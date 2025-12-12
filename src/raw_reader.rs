// Copyright 2024 Cornell University
// Copyright 2025 Shareef Jalloq
// released under BSD 3-Clause License
// Low-level raw access to FST file components for filtered copying.
//
// This module provides APIs that allow reading FST components with minimal
// decompression, enabling efficient filtered copies by copying compressed
// value change data directly.

use crate::io::{read_signal_locs, read_u64, read_u8, read_variant_u64, OffsetTable, ReaderError};
use crate::reader::{BufReadSeekAny, FstReader, Result};
use crate::types::DataSectionKind;
use std::io::{BufRead, Seek, SeekFrom, Write};

/// Information about a VC (Value Change) block's location and time range.
#[derive(Debug, Clone)]
pub struct VcBlockInfo {
    /// Index of this block (0-based)
    pub index: usize,
    /// File offset where the block starts
    pub file_offset: u64,
    /// Start time of this block's time window
    pub start_time: u64,
    /// End time of this block's time window
    pub end_time: u64,
    /// Block type (Standard, DynamicAlias, DynamicAlias2)
    pub kind: VcBlockKind,
    /// Memory required for full traversal (hint from file)
    pub mem_required: u64,
}

/// The type of VC block (determines position table format)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VcBlockKind {
    Standard,
    DynamicAlias,
    DynamicAlias2,
}

impl From<DataSectionKind> for VcBlockKind {
    fn from(k: DataSectionKind) -> Self {
        match k {
            DataSectionKind::Standard => VcBlockKind::Standard,
            DataSectionKind::DynamicAlias => VcBlockKind::DynamicAlias,
            DataSectionKind::DynamicAlias2 => VcBlockKind::DynamicAlias2,
        }
    }
}

impl From<VcBlockKind> for DataSectionKind {
    fn from(k: VcBlockKind) -> Self {
        match k {
            VcBlockKind::Standard => DataSectionKind::Standard,
            VcBlockKind::DynamicAlias => DataSectionKind::DynamicAlias,
            VcBlockKind::DynamicAlias2 => DataSectionKind::DynamicAlias2,
        }
    }
}

/// Location of a signal's compressed value change data within a VC block.
#[derive(Debug, Clone, Copy)]
pub struct SignalDataLocation {
    /// Signal index
    pub signal_idx: usize,
    /// Byte offset from the start of the waves data section
    pub offset: u64,
    /// Length of the compressed data in bytes
    pub length: u32,
}

/// Position table mapping signal indices to their data locations.
/// Wraps the internal OffsetTable and provides iteration over signals with data.
pub struct PositionTable {
    inner: OffsetTable,
}

impl PositionTable {
    /// Iterate over signals that have data (skipping None entries, resolving aliases).
    pub fn iter(&self) -> impl Iterator<Item = SignalDataLocation> + '_ {
        self.inner.iter().map(|entry| SignalDataLocation {
            signal_idx: entry.signal_idx,
            offset: entry.offset,
            length: entry.len,
        })
    }
}

/// Compression type for value change data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VcPackType {
    /// LZ4 compression
    Lz4,
    /// FastLZ compression
    FastLz,
    /// ZLib compression
    Zlib,
}

/// Raw reader for accessing VC block components with minimal decompression.
pub struct VcBlockReader<'a> {
    input: &'a mut dyn BufReadSeekAny,
    info: VcBlockInfo,
    section_length: u64,
    waves_data_start: Option<u64>,
    pack_type: Option<VcPackType>,
}

impl<'a> VcBlockReader<'a> {
    /// Create a new VcBlockReader for the given block.
    pub(crate) fn new(input: &'a mut dyn BufReadSeekAny, info: VcBlockInfo) -> Result<Self> {
        // Read section length
        input.seek(SeekFrom::Start(info.file_offset))?;
        let section_length = read_u64(input)?;

        Ok(Self {
            input,
            info,
            section_length,
            waves_data_start: None,
            pack_type: None,
        })
    }

    /// Get information about this block.
    pub fn info(&self) -> &VcBlockInfo {
        &self.info
    }

    /// Get the section length.
    pub fn section_length(&self) -> u64 {
        self.section_length
    }

    /// Read and decompress the position table for this block.
    /// This must be called before copy_signal_data or read_signal_data_raw.
    pub fn read_position_table(&mut self) -> Result<PositionTable> {
        // Read time section info to find position table location
        let time_section_length = self.read_time_section_length()?;

        // Position table (chain) is right before time section
        let chain_len_offset =
            self.info.file_offset + self.section_length - time_section_length - 8;

        // Read frame header to find waves data start
        let frame_start = self.info.file_offset + 4 * 8;
        self.input.seek(SeekFrom::Start(frame_start))?;

        let (uncompressed_len, _) = read_variant_u64(self.input)?;
        let (compressed_len, _) = read_variant_u64(self.input)?;
        let (max_handle, _) = read_variant_u64(self.input)?;

        // Skip the frame data
        let frame_data_len = if compressed_len > 0 {
            compressed_len
        } else {
            uncompressed_len
        };
        self.input.seek(SeekFrom::Current(frame_data_len as i64))?;

        // Now we're at the start of value change data
        let (_max_handle2, _) = read_variant_u64(self.input)?;
        let vc_start = self.input.stream_position()?;
        let pack_type_byte = read_u8(self.input)?;
        let pack_type = match pack_type_byte {
            b'4' => VcPackType::Lz4,
            b'F' => VcPackType::FastLz,
            _ => VcPackType::Zlib,
        };

        self.waves_data_start = Some(vc_start);
        self.pack_type = Some(pack_type);

        // Read position table
        let section_kind: DataSectionKind = self.info.kind.into();

        let offset_table = read_signal_locs(
            self.input,
            chain_len_offset,
            section_kind,
            max_handle,
            vc_start,
        )?;

        Ok(PositionTable { inner: offset_table })
    }

    /// Copy the compressed value change data for a signal to a writer.
    /// Returns the number of bytes copied.
    /// Must call read_position_table first.
    pub fn copy_signal_data<W: Write>(
        &mut self,
        loc: &SignalDataLocation,
        out: &mut W,
    ) -> Result<u32> {
        let waves_start = self.waves_data_start.ok_or_else(|| {
            ReaderError::Io(std::io::Error::other(
                "must call read_position_table before copy_signal_data",
            ))
        })?;

        // Seek to the signal's data
        self.input.seek(SeekFrom::Start(waves_start + loc.offset))?;

        // Copy the compressed bytes
        let mut buf = vec![0u8; loc.length as usize];
        self.input.read_exact(&mut buf)?;
        out.write_all(&buf)?;

        Ok(loc.length)
    }

    /// Read the raw compressed bytes for a signal's value changes.
    /// Must call read_position_table first.
    pub fn read_signal_data_raw(&mut self, loc: &SignalDataLocation) -> Result<Vec<u8>> {
        let waves_start = self.waves_data_start.ok_or_else(|| {
            ReaderError::Io(std::io::Error::other(
                "must call read_position_table before read_signal_data_raw",
            ))
        })?;

        self.input.seek(SeekFrom::Start(waves_start + loc.offset))?;
        let mut buf = vec![0u8; loc.length as usize];
        self.input.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Get the compression type used for value changes in this block.
    pub fn pack_type(&self) -> Option<VcPackType> {
        self.pack_type
    }

    /// Copy the entire VC block as raw bytes (for verbatim copy when no filtering needed).
    /// This copies from the block type byte through the entire section.
    pub fn copy_raw_block<W: Write>(&mut self, out: &mut W) -> Result<u64> {
        // Seek to start of block (before section length)
        self.input.seek(SeekFrom::Start(self.info.file_offset - 1))?;

        // Write block type
        let block_type_byte = match self.info.kind {
            VcBlockKind::Standard => 1u8,
            VcBlockKind::DynamicAlias => 5u8,
            VcBlockKind::DynamicAlias2 => 8u8,
        };
        out.write_all(&[block_type_byte])?;

        // Seek to section length
        self.input.seek(SeekFrom::Start(self.info.file_offset))?;

        // Copy entire block including length field
        let total_len = self.section_length;
        let mut remaining = total_len as usize;
        let mut buf = vec![0u8; 64 * 1024]; // 64KB buffer

        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            self.input.read_exact(&mut buf[..to_read])?;
            out.write_all(&buf[..to_read])?;
            remaining -= to_read;
        }

        Ok(total_len + 1) // +1 for block type byte
    }

    // Internal: read time section length from end of block
    fn read_time_section_length(&mut self) -> Result<u64> {
        // Time table info is at the end of the section
        // Structure: ... | time_data | uncompressed_len (8) | compressed_len (8) | time_count (8) |
        let time_info_offset = self.info.file_offset + self.section_length - 3 * 8;
        self.input.seek(SeekFrom::Start(time_info_offset))?;

        let _uncompressed_len = read_u64(self.input)?;
        let compressed_len = read_u64(self.input)?;
        let _time_count = read_u64(self.input)?;

        // Time section includes the data + 3 u64s of metadata
        Ok(compressed_len + 3 * 8)
    }
}

/// Extension methods for FstReader to provide raw access to VC blocks.
impl<R: BufRead + Seek> FstReader<R> {
    /// Get information about all VC blocks in the file.
    pub fn vc_block_infos(&self) -> Vec<VcBlockInfo> {
        self.get_data_sections()
            .iter()
            .enumerate()
            .map(|(idx, section)| VcBlockInfo {
                index: idx,
                file_offset: section.file_offset,
                start_time: section.start_time,
                end_time: section.end_time,
                kind: section.kind.into(),
                mem_required: section.mem_required_for_traversal,
            })
            .collect()
    }

    /// Get the number of VC blocks in the file.
    pub fn vc_block_count(&self) -> usize {
        self.get_data_sections().len()
    }

    /// Check if a block's time range overlaps with the given range.
    pub fn block_overlaps_time(&self, block_idx: usize, start: u64, end: u64) -> bool {
        if let Some(section) = self.get_data_sections().get(block_idx) {
            section.end_time >= start && section.start_time <= end
        } else {
            false
        }
    }

    /// Check if a block is entirely within the given time range.
    pub fn block_within_time(&self, block_idx: usize, start: u64, end: u64) -> bool {
        if let Some(section) = self.get_data_sections().get(block_idx) {
            section.start_time >= start && section.end_time <= end
        } else {
            false
        }
    }

    /// Execute a callback with a raw VC block reader.
    /// This provides raw access to the block's position table and signal data.
    ///
    /// # Example
    /// ```ignore
    /// let block_infos = reader.vc_block_infos();
    /// reader.with_vc_block(&block_infos[0], |block_reader| {
    ///     let pos_table = block_reader.read_position_table()?;
    ///     for loc in pos_table.iter() {
    ///         let data = block_reader.read_signal_data_raw(&loc)?;
    ///         // process raw data...
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn with_vc_block<T, F>(&mut self, block_info: &VcBlockInfo, f: F) -> Result<T>
    where
        F: FnOnce(&mut VcBlockReader<'_>) -> Result<T>,
    {
        let info = block_info.clone();
        self.with_input_mut(|input| {
            let mut reader = VcBlockReader::new(input, info)?;
            f(&mut reader)
        })
    }

    /// Read and decompress the frame (initial values) from a VC block.
    ///
    /// The frame contains the initial values for all signals at the start of the block.
    /// Returns the decompressed frame bytes.
    pub fn read_frame_raw(&mut self, block_info: &VcBlockInfo) -> Result<Vec<u8>> {
        use crate::io::{read_bytes, read_variant_u64, read_zlib_compressed_bytes};

        self.with_input_mut(|input| {
            // Seek to frame header (after section header: length, start, end, mem_req)
            let frame_start = block_info.file_offset + 4 * 8;
            input.seek(SeekFrom::Start(frame_start))?;

            let (uncompressed_len, _) = read_variant_u64(input)?;
            let (compressed_len, _) = read_variant_u64(input)?;
            let (_max_handle, _) = read_variant_u64(input)?;

            // Read frame data
            if compressed_len > 0 {
                read_zlib_compressed_bytes(input, uncompressed_len, compressed_len, true)
            } else {
                read_bytes(input, uncompressed_len as usize)
            }
        })
    }

    /// Read the raw time table data from a VC block.
    ///
    /// Returns (compressed_data, uncompressed_len, compressed_len, time_count).
    /// The time data can be copied verbatim to a new FST file.
    pub fn read_time_table_raw(
        &mut self,
        block_info: &VcBlockInfo,
    ) -> Result<(Vec<u8>, u64, u64, u64)> {
        use crate::io::{read_bytes, read_u64};

        self.with_input_mut(|input| {
            // Read section length
            input.seek(SeekFrom::Start(block_info.file_offset))?;
            let section_length = read_u64(input)?;

            // Time table footer is at end of section
            let time_footer_offset = block_info.file_offset + section_length - 3 * 8;
            input.seek(SeekFrom::Start(time_footer_offset))?;

            let uncompressed_len = read_u64(input)?;
            let compressed_len = read_u64(input)?;
            let time_count = read_u64(input)?;

            // Read compressed time data
            let time_data_offset = time_footer_offset - compressed_len;
            input.seek(SeekFrom::Start(time_data_offset))?;
            let time_data = read_bytes(input, compressed_len as usize)?;

            Ok((time_data, uncompressed_len, compressed_len, time_count))
        })
    }
}
