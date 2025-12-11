//! FST filter and inspection tool
//!
//! A command-line tool for inspecting and filtering FST waveform files.
//! Supports displaying metadata, browsing hierarchy, and creating filtered
//! copies with optimized raw block copying (no decompression of value changes).
//!
//! # Examples
//!
//! Display file metadata:
//! ```sh
//! cargo run --release --example fst-filter -- input.fst --info
//! ```
//!
//! List hierarchy to a certain depth:
//! ```sh
//! cargo run --release --example fst-filter -- input.fst --hierarchy --max-depth 3
//! ```
//!
//! Filter signals by hierarchy path:
//! ```sh
//! cargo run --release --example fst-filter -- input.fst -o output.fst top.cpu.alu
//! ```
//!
//! # Performance Optimizations
//!
//! This tool is optimized for processing large FST files (multi-GB with millions
//! of signals):
//!
//! - **FxHashMap**: Uses rustc-hash's FxHashMap instead of std::collections::HashMap
//!   for faster hashing of integer keys (~18% speedup)
//! - **Raw block copying**: Value change data is copied without decompression
//!   when possible, preserving the original compression
//! - **Signal deduplication**: Signals sharing the same data (aliases) are
//!   detected and written once, with position table aliases pointing to the
//!   shared data

use clap::Parser;
use fst_reader::{FstHierarchyEntry, FstReader, FstScopeType, FstVarDirection, FstVarType, VcPackType as ReaderPackType};
use fst_writer::{extract_filtered_frame, SignalGeometry, VcBlockWriter, VcPackType};
use glob::Pattern;
use regex::Regex;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};

#[derive(Parser)]
#[command(name = "fst-filter")]
#[command(about = "FST waveform file inspection and filtering tool")]
#[command(version)]
struct Cli {
    /// Input FST file
    input: String,

    /// Output FST file (required for filtering)
    #[arg(short, long)]
    output: Option<String>,

    /// Display file metadata (header information)
    #[arg(long)]
    info: bool,

    /// Display hierarchy tree
    #[arg(long)]
    hierarchy: bool,

    /// Maximum depth for hierarchy display (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_depth: usize,

    /// Include signals in hierarchy display (default shows only scopes)
    #[arg(long)]
    with_signals: bool,

    /// Show hierarchy as flat paths (for copy/paste into filter)
    #[arg(long)]
    flat: bool,

    /// Hierarchy paths to include (substring match by default)
    #[arg(value_name = "PATH")]
    filter_paths: Vec<String>,

    /// Treat filter paths as glob patterns
    #[arg(short = 'g', long)]
    glob: bool,

    /// Treat filter paths as regex patterns
    #[arg(short = 'e', long)]
    regex: bool,

    /// Quiet mode - minimal output
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Open input file
    let input = File::open(&cli.input)?;
    let mut reader = FstReader::open_and_read_time_table(BufReader::new(input))?;

    // Handle info display
    if cli.info {
        display_info(&reader);
        if !cli.hierarchy && cli.filter_paths.is_empty() {
            return Ok(());
        }
    }

    // Handle hierarchy display
    if cli.hierarchy {
        display_hierarchy(&mut reader, &cli)?;
        // When --hierarchy is used, filter_paths are for filtering the display, not for creating output
        return Ok(());
    }

    // Handle filtering (create filtered output file)
    if !cli.filter_paths.is_empty() {
        if cli.output.is_none() {
            eprintln!("Error: --output is required when filtering signals");
            std::process::exit(1);
        }
        filter_fst(&mut reader, &cli)?;
    }

    // Default: show info if no action specified
    if !cli.info && !cli.hierarchy && cli.filter_paths.is_empty() {
        display_info(&reader);
    }

    Ok(())
}

fn display_info<R: std::io::BufRead + std::io::Seek>(reader: &FstReader<R>) {
    let header = reader.get_header();
    let timescale_exp = header.timescale_exponent;

    println!("FST File Information");
    println!("====================");
    println!("Version:      {}", header.version);
    println!("Date:         {}", header.date);
    println!("Timescale:    {}", format_timescale(timescale_exp));
    println!("Start time:   {} ({})", header.start_time, format_time(header.start_time, timescale_exp));
    println!("End time:     {} ({})", header.end_time, format_time(header.end_time, timescale_exp));
    println!("Duration:     {}", format_time(header.end_time - header.start_time, timescale_exp));
    println!("Variables:    {}", header.var_count);
    println!("Signals:      {}", reader.signal_count());
    println!("VC blocks:    {}", reader.vc_block_count());

    // Display VC block info
    let blocks = reader.vc_block_infos();
    if !blocks.is_empty() {
        println!("\nVC Block Summary:");
        for (i, block) in blocks.iter().enumerate() {
            println!(
                "  Block {:3}: {} - {} (offset 0x{:08x})",
                i,
                format_time(block.start_time, timescale_exp),
                format_time(block.end_time, timescale_exp),
                block.file_offset
            );
        }
    }
}

fn display_hierarchy<R: std::io::BufRead + std::io::Seek>(
    reader: &mut FstReader<R>,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build filter matchers from filter_paths
    let glob_patterns: Option<Vec<Pattern>> = if cli.glob && !cli.filter_paths.is_empty() {
        Some(
            cli.filter_paths
                .iter()
                .map(|p| Pattern::new(p))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };

    let regex_patterns: Option<Vec<Regex>> = if cli.regex && !cli.filter_paths.is_empty() {
        Some(
            cli.filter_paths
                .iter()
                .map(|p| Regex::new(p))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };

    let mut scope_stack: Vec<String> = Vec::new();
    let mut signal_count = 0;
    let mut scope_count = 0;

    if !cli.flat {
        println!("\nHierarchy:");
        println!("----------");
    }

    reader.read_hierarchy(|entry| {
        match entry {
            FstHierarchyEntry::Scope { name, tpe, .. } => {
                scope_stack.push(name.clone());
                let depth = scope_stack.len();
                let full_path = scope_stack.join(".");

                // Check depth limit
                if cli.max_depth > 0 && depth > cli.max_depth {
                    return;
                }

                // Check filters (if any specified)
                if !cli.filter_paths.is_empty() {
                    if !path_matches_filters(&full_path, &cli.filter_paths, &glob_patterns, &regex_patterns) {
                        return;
                    }
                }

                if cli.flat {
                    println!("{}", full_path);
                } else {
                    let indent = "  ".repeat(depth - 1);
                    println!("{}{} ({})", indent, name, scope_type_str(&tpe));
                }
                scope_count += 1;
            }
            FstHierarchyEntry::UpScope => {
                scope_stack.pop();
            }
            FstHierarchyEntry::Var {
                name,
                tpe,
                direction,
                length,
                is_alias,
                ..
            } => {
                // Only show signals if --with-signals is specified
                if !cli.with_signals {
                    return;
                }

                if is_alias {
                    return;
                }

                let depth = scope_stack.len() + 1;
                let full_path = if scope_stack.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", scope_stack.join("."), name)
                };

                // Check depth limit
                if cli.max_depth > 0 && depth > cli.max_depth {
                    return;
                }

                // Check filters (if any specified)
                if !cli.filter_paths.is_empty() {
                    if !path_matches_filters(&full_path, &cli.filter_paths, &glob_patterns, &regex_patterns) {
                        return;
                    }
                }

                if cli.flat {
                    println!("{}", full_path);
                } else {
                    let indent = "  ".repeat(depth - 1);
                    let dir_str = direction_str(&direction);
                    let type_str = var_type_str(&tpe);
                    if length > 1 {
                        println!("{}{}[{}:0] {} {}", indent, name, length - 1, type_str, dir_str);
                    } else {
                        println!("{}{} {} {}", indent, name, type_str, dir_str);
                    }
                }
                signal_count += 1;
            }
            _ => {}
        }
    })?;

    if !cli.flat {
        println!("\nTotal: {} scopes, {} signals", scope_count, signal_count);
    }
    Ok(())
}

/// Check if a path matches any of the filter patterns
fn path_matches_filters(
    full_path: &str,
    paths: &[String],
    globs: &Option<Vec<Pattern>>,
    regexes: &Option<Vec<Regex>>,
) -> bool {
    if let Some(patterns) = globs {
        return patterns.iter().any(|p| p.matches(full_path));
    }

    if let Some(patterns) = regexes {
        return patterns.iter().any(|r| r.is_match(full_path));
    }

    // Default: substring match
    paths.iter().any(|p| full_path.contains(p))
}

fn filter_fst<R: std::io::BufRead + std::io::Seek>(
    reader: &mut FstReader<R>,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = cli.output.as_ref().unwrap();

    // Build filter matchers
    let glob_patterns: Option<Vec<Pattern>> = if cli.glob {
        Some(
            cli.filter_paths
                .iter()
                .map(|p| Pattern::new(p))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };

    let regex_patterns: Option<Vec<Regex>> = if cli.regex {
        Some(
            cli.filter_paths
                .iter()
                .map(|p| Regex::new(p))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };

    // Read hierarchy and identify signals to keep
    let mut signals_to_keep: HashSet<usize> = HashSet::new();
    let mut scope_stack: Vec<String> = Vec::new();
    let mut hierarchy_entries: Vec<FstHierarchyEntry> = Vec::new();

    reader.read_hierarchy(|entry| {
        hierarchy_entries.push(entry.clone());
        match &entry {
            FstHierarchyEntry::Scope { name, .. } => {
                scope_stack.push(name.clone());
            }
            FstHierarchyEntry::UpScope => {
                scope_stack.pop();
            }
            FstHierarchyEntry::Var {
                name,
                handle,
                is_alias,
                ..
            } => {
                if *is_alias {
                    return;
                }

                let full_path = if scope_stack.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", scope_stack.join("."), name)
                };

                let include = path_matches_filters(
                    &full_path,
                    &cli.filter_paths,
                    &glob_patterns,
                    &regex_patterns,
                );

                if include {
                    signals_to_keep.insert(handle.get_index());
                }
            }
            _ => {}
        }
    })?;

    // Get signal geometries
    let geom_tuples = reader.signal_geometries();
    let geometries: Vec<SignalGeometry> = geom_tuples
        .iter()
        .map(|&(frame_bytes, is_real)| SignalGeometry { frame_bytes, is_real })
        .collect();

    // Sort kept signals
    let mut kept_signals: Vec<usize> = signals_to_keep.into_iter().collect();
    kept_signals.sort();

    if kept_signals.is_empty() {
        eprintln!("Error: No signals match the filter");
        std::process::exit(1);
    }

    if !cli.quiet {
        println!("Filtering {} signals to {}", kept_signals.len(), output_path);
    }

    // Create output file and write
    let header = reader.get_header();
    let output = File::create(output_path)?;
    let mut writer = BufWriter::new(output);

    // Write header
    write_fst_header(
        &mut writer,
        header.start_time,
        header.end_time,
        kept_signals.len() as u64,
        &header.version,
        &header.date,
        header.timescale_exponent,
    )?;

    // Copy VC blocks
    let block_infos = reader.vc_block_infos();

    for block_info in block_infos.iter() {
        let source_frame = reader.read_frame_raw(block_info)?;

        let (time_data, time_uncomp, time_comp, time_count) =
            reader.read_time_table_raw(block_info)?;

        // Maps (offset, length) in source -> raw compressed data bytes.
        // Using FxHashMap for faster integer key hashing.
        let mut source_offset_to_data: FxHashMap<(u64, u32), Vec<u8>> = FxHashMap::default();
        // Maps signal index -> its (offset, length) key in the source file
        let mut signal_to_source_key: FxHashMap<usize, (u64, u32)> = FxHashMap::default();
        let mut pack_type = VcPackType::Lz4;

        reader.with_vc_block(block_info, |block_reader| {
            let pos_table = block_reader.read_position_table()?;
            pack_type = match block_reader.pack_type().unwrap_or(ReaderPackType::Lz4) {
                ReaderPackType::Lz4 => VcPackType::Lz4,
                ReaderPackType::FastLz => VcPackType::FastLz,
                ReaderPackType::Zlib => VcPackType::Zlib,
            };

            // Build lookup map for O(1) access to signal locations.
            // FxHashMap provides ~18% speedup over std HashMap for integer keys.
            let mut loc_map: FxHashMap<usize, _> = FxHashMap::default();
            for loc in pos_table.iter() {
                loc_map.insert(loc.signal_idx, loc);
            }

            // Read raw (still compressed) data for each kept signal.
            // Deduplication: signals sharing the same (offset, length) in the
            // source file share the same data blob - we only read it once.
            for &sig_idx in &kept_signals {
                if let Some(loc) = loc_map.get(&sig_idx) {
                    let key = (loc.offset, loc.length);
                    signal_to_source_key.insert(sig_idx, key);
                    if !source_offset_to_data.contains_key(&key) {
                        let data = block_reader.read_signal_data_raw(loc)?;
                        source_offset_to_data.insert(key, data);
                    }
                }
            }
            Ok(())
        })?;

        let filtered_frame = extract_filtered_frame(&geometries, &kept_signals, &source_frame);
        let mut block_writer =
            VcBlockWriter::new(&mut writer, block_info.start_time, block_info.end_time)?;

        block_writer.write_frame(&filtered_frame, kept_signals.len())?;
        let _waves_start = block_writer.begin_waves(kept_signals.len(), pack_type)?;

        // Write signal data with deduplication.
        // Multiple signals may share the same underlying data (aliases in the source).
        // We track which data blobs have been written and reuse offsets for duplicates.
        let mut source_to_output_offset: FxHashMap<(u64, u32), u64> = FxHashMap::default();
        let mut signal_offsets: Vec<(bool, u64)> = Vec::new();
        let mut current_offset = 1u64; // FST offsets start at 1 (0 means no data)

        for &sig_idx in &kept_signals {
            if let Some(&key) = signal_to_source_key.get(&sig_idx) {
                if let Some(&out_offset) = source_to_output_offset.get(&key) {
                    // Data already written - record alias to existing offset
                    signal_offsets.push((true, out_offset));
                } else {
                    // First occurrence of this data blob - write it
                    let data = source_offset_to_data.get(&key).unwrap();
                    signal_offsets.push((true, current_offset));
                    source_to_output_offset.insert(key, current_offset);
                    block_writer.write_raw_wave_data(data)?;
                    current_offset += data.len() as u64;
                }
            } else {
                // Signal has no value change data in this block
                signal_offsets.push((false, 0));
            }
        }

        block_writer.write_position_table(&signal_offsets)?;
        block_writer.write_time_table_raw(&time_data, time_uncomp, time_comp, time_count)?;

        let block_waves: u64 = source_offset_to_data.values().map(|d| d.len() as u64).sum();
        let mem_required = filtered_frame.len() as u64 + block_waves;
        block_writer.finish(mem_required)?;
    }

    // Write geometry and hierarchy
    write_geometry_block(&mut writer, &kept_signals, &geometries)?;

    let kept_set: HashSet<usize> = kept_signals.iter().copied().collect();
    write_filtered_hierarchy(&mut writer, &hierarchy_entries, &kept_set)?;

    writer.flush()?;

    if !cli.quiet {
        let input_size = std::fs::metadata(&cli.input)?.len();
        let output_size = std::fs::metadata(output_path)?.len();
        println!(
            "Complete: {} -> {} ({:.1}% of original)",
            format_size(input_size),
            format_size(output_size),
            output_size as f64 / input_size as f64 * 100.0
        );
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format timescale exponent as a human-readable string
fn format_timescale(exp: i8) -> String {
    match exp {
        0 => "1 s".to_string(),
        -1 => "100 ms".to_string(),
        -2 => "10 ms".to_string(),
        -3 => "1 ms".to_string(),
        -4 => "100 us".to_string(),
        -5 => "10 us".to_string(),
        -6 => "1 us".to_string(),
        -7 => "100 ns".to_string(),
        -8 => "10 ns".to_string(),
        -9 => "1 ns".to_string(),
        -10 => "100 ps".to_string(),
        -11 => "10 ps".to_string(),
        -12 => "1 ps".to_string(),
        -13 => "100 fs".to_string(),
        -14 => "10 fs".to_string(),
        -15 => "1 fs".to_string(),
        _ => format!("10^{} s", exp),
    }
}

/// Format a time value (in timescale units) as a human-readable string with SI units
fn format_time(time: u64, timescale_exp: i8) -> String {
    // Convert time to seconds (as f64)
    let seconds = time as f64 * 10_f64.powi(timescale_exp as i32);

    // Choose appropriate unit
    if seconds == 0.0 {
        "0 s".to_string()
    } else if seconds >= 1.0 {
        format!("{:.3} s", seconds)
    } else if seconds >= 1e-3 {
        format!("{:.3} ms", seconds * 1e3)
    } else if seconds >= 1e-6 {
        format!("{:.3} us", seconds * 1e6)
    } else if seconds >= 1e-9 {
        format!("{:.3} ns", seconds * 1e9)
    } else if seconds >= 1e-12 {
        format!("{:.3} ps", seconds * 1e12)
    } else {
        format!("{:.3} fs", seconds * 1e15)
    }
}

fn scope_type_str(tpe: &FstScopeType) -> &'static str {
    use FstScopeType::*;
    match tpe {
        Module => "module",
        Task => "task",
        Function => "function",
        Begin => "begin",
        Fork => "fork",
        Generate => "generate",
        Struct => "struct",
        Union => "union",
        Class => "class",
        Interface => "interface",
        Package => "package",
        Program => "program",
        VhdlArchitecture => "architecture",
        VhdlProcedure => "procedure",
        VhdlFunction => "vhdl_function",
        VhdlRecord => "record",
        VhdlProcess => "process",
        VhdlBlock => "block",
        VhdlForGenerate => "for_generate",
        VhdlIfGenerate => "if_generate",
        VhdlGenerate => "generate",
        VhdlPackage => "vhdl_package",
        _ => "scope",
    }
}

fn var_type_str(tpe: &FstVarType) -> &'static str {
    use FstVarType::*;
    match tpe {
        Wire => "wire",
        Reg => "reg",
        Logic => "logic",
        Integer => "integer",
        Real => "real",
        Parameter => "parameter",
        Event => "event",
        Time => "time",
        Bit => "bit",
        Int => "int",
        ShortInt => "shortint",
        LongInt => "longint",
        Byte => "byte",
        Enum => "enum",
        _ => "var",
    }
}

fn direction_str(dir: &FstVarDirection) -> &'static str {
    use FstVarDirection::*;
    match dir {
        Implicit => "",
        Input => "input",
        Output => "output",
        InOut => "inout",
        Buffer => "buffer",
        Linkage => "linkage",
    }
}

// ============================================================================
// FST writing helper functions
// ============================================================================

fn write_fst_header<W: Write + Seek>(
    output: &mut W,
    start_time: u64,
    end_time: u64,
    var_count: u64,
    version: &str,
    date: &str,
    timescale_exp: i8,
) -> std::io::Result<()> {
    const HEADER_LENGTH: u64 = 329;
    const FST_BL_HDR: u8 = 0;

    output.write_all(&[FST_BL_HDR])?;
    output.write_all(&HEADER_LENGTH.to_be_bytes())?;
    output.write_all(&start_time.to_be_bytes())?;
    output.write_all(&end_time.to_be_bytes())?;

    let endian_test: f64 = std::f64::consts::E;
    output.write_all(&endian_test.to_le_bytes())?;

    output.write_all(&0u64.to_be_bytes())?; // memory used
    output.write_all(&0u64.to_be_bytes())?; // scope count
    output.write_all(&var_count.to_be_bytes())?;
    output.write_all(&var_count.to_be_bytes())?; // max var ID length
    output.write_all(&1u64.to_be_bytes())?; // VC section count
    output.write_all(&[timescale_exp as u8])?;

    let mut version_buf = [0u8; 128];
    let copy_len = version.len().min(127);
    version_buf[..copy_len].copy_from_slice(&version.as_bytes()[..copy_len]);
    output.write_all(&version_buf)?;

    let mut date_buf = [0u8; 119];
    let copy_len = date.len().min(118);
    date_buf[..copy_len].copy_from_slice(&date.as_bytes()[..copy_len]);
    output.write_all(&date_buf)?;

    output.write_all(&[0u8])?; // file type
    output.write_all(&0i64.to_be_bytes())?; // time zero

    Ok(())
}

fn write_geometry_block<W: Write + Seek>(
    output: &mut W,
    kept_signals: &[usize],
    geometries: &[SignalGeometry],
) -> std::io::Result<()> {
    const FST_BL_GEOM: u8 = 3;

    let mut geom_data = Vec::new();
    for &sig_idx in kept_signals {
        let g = &geometries[sig_idx];
        let value = if g.is_real {
            0u32
        } else if g.frame_bytes == 0 {
            u32::MAX
        } else {
            g.frame_bytes
        };
        write_varint(&mut geom_data, value as u64);
    }

    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&geom_data, 4);
    let data = if compressed.len() < geom_data.len() {
        &compressed[..]
    } else {
        &geom_data[..]
    };

    output.write_all(&[FST_BL_GEOM])?;
    let section_len = 3 * 8 + data.len() as u64;
    output.write_all(&section_len.to_be_bytes())?;
    output.write_all(&(geom_data.len() as u64).to_be_bytes())?;
    output.write_all(&(kept_signals.len() as u64).to_be_bytes())?;
    output.write_all(data)?;

    Ok(())
}

fn write_filtered_hierarchy<W: Write + Seek>(
    output: &mut W,
    hierarchy_entries: &[FstHierarchyEntry],
    kept_signals: &HashSet<usize>,
) -> std::io::Result<()> {
    const FST_BL_HIER: u8 = 4;
    const FST_ST_VCD_SCOPE: u8 = 254;
    const FST_ST_VCD_UPSCOPE: u8 = 255;
    const GZIP_HEADER: [u8; 10] = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255];

    let mut hier_data = Vec::new();
    let mut scope_stack: Vec<(String, u8, String)> = Vec::new();
    let mut scopes_written: usize = 0;

    for entry in hierarchy_entries {
        match entry {
            FstHierarchyEntry::Scope { name, tpe, component } => {
                scope_stack.push((name.clone(), scope_type_to_u8(*tpe), component.clone()));
            }
            FstHierarchyEntry::UpScope => {
                if scopes_written >= scope_stack.len() && scopes_written > 0 {
                    hier_data.push(FST_ST_VCD_UPSCOPE);
                    scopes_written -= 1;
                }
                scope_stack.pop();
            }
            FstHierarchyEntry::Var {
                handle,
                is_alias,
                name,
                tpe,
                direction,
                length,
                ..
            } => {
                if *is_alias {
                    continue;
                }

                let handle_idx = handle.get_index();
                if !kept_signals.contains(&handle_idx) {
                    continue;
                }

                while scopes_written < scope_stack.len() {
                    let (scope_name, scope_type, component) = &scope_stack[scopes_written];
                    hier_data.push(FST_ST_VCD_SCOPE);
                    hier_data.push(*scope_type);
                    hier_data.extend_from_slice(scope_name.as_bytes());
                    hier_data.push(0x00);
                    hier_data.extend_from_slice(component.as_bytes());
                    hier_data.push(0x00);
                    scopes_written += 1;
                }

                hier_data.push(var_type_to_u8(*tpe));
                hier_data.push(var_direction_to_u8(*direction));
                hier_data.extend_from_slice(name.as_bytes());
                hier_data.push(0x00);
                write_varint(&mut hier_data, *length as u64);
                write_varint(&mut hier_data, 0); // alias
            }
            _ => {}
        }
    }

    while scopes_written > 0 {
        hier_data.push(FST_ST_VCD_UPSCOPE);
        scopes_written -= 1;
    }

    let compressed = miniz_oxide::deflate::compress_to_vec(&hier_data, 4);

    output.write_all(&[FST_BL_HIER])?;
    let section_len = 16 + GZIP_HEADER.len() as u64 + compressed.len() as u64;
    output.write_all(&section_len.to_be_bytes())?;
    output.write_all(&(hier_data.len() as u64).to_be_bytes())?;
    output.write_all(&GZIP_HEADER)?;
    output.write_all(&compressed)?;

    Ok(())
}

fn scope_type_to_u8(tpe: FstScopeType) -> u8 {
    use FstScopeType::*;
    match tpe {
        Module => 0,
        Task => 1,
        Function => 2,
        Begin => 3,
        Fork => 4,
        Generate => 5,
        Struct => 6,
        Union => 7,
        Class => 8,
        Interface => 9,
        Package => 10,
        Program => 11,
        VhdlArchitecture => 12,
        VhdlProcedure => 13,
        VhdlFunction => 14,
        VhdlRecord => 15,
        VhdlProcess => 16,
        VhdlBlock => 17,
        VhdlForGenerate => 18,
        VhdlIfGenerate => 19,
        VhdlGenerate => 20,
        VhdlPackage => 21,
        _ => 0,
    }
}

fn var_type_to_u8(tpe: FstVarType) -> u8 {
    use FstVarType::*;
    match tpe {
        Event => 0,
        Integer => 1,
        Parameter => 2,
        Real => 3,
        RealParameter => 4,
        Reg => 5,
        Supply0 => 6,
        Supply1 => 7,
        Time => 8,
        Tri => 9,
        TriAnd => 10,
        TriOr => 11,
        TriReg => 12,
        Tri0 => 13,
        Tri1 => 14,
        Wand => 15,
        Wire => 16,
        Wor => 17,
        Port => 18,
        SparseArray => 19,
        RealTime => 20,
        GenericString => 21,
        Bit => 22,
        Logic => 23,
        Int => 24,
        ShortInt => 25,
        LongInt => 26,
        Byte => 27,
        Enum => 28,
        ShortReal => 29,
    }
}

fn var_direction_to_u8(dir: FstVarDirection) -> u8 {
    use FstVarDirection::*;
    match dir {
        Implicit => 0,
        Input => 1,
        Output => 2,
        InOut => 3,
        Buffer => 4,
        Linkage => 5,
    }
}

fn write_varint(output: &mut Vec<u8>, mut value: u64) {
    if value <= 0x7f {
        output.push(value as u8);
        return;
    }
    while value != 0 {
        let next_value = value >> 7;
        let mask: u8 = if next_value == 0 { 0 } else { 0x80 };
        output.push((value & 0x7f) as u8 | mask);
        value = next_value;
    }
}
