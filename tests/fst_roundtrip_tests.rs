// Tests for FST write -> read roundtrips
//
// These tests verify that FST files written by fst-writer can be read back
// by fst-reader, catching bugs like incorrect section_length encoding.

use fst_reader::*;
use fst_writer::{
    open_fst, FstFileType, FstInfo, FstScopeType, FstSignalType, FstVarDirection, FstVarType,
};
use std::io::BufReader;

/// Create a simple FST file and return its path.
///
/// Uses a temp file that will be cleaned up when the test ends.
fn create_simple_fst() -> tempfile::NamedTempFile {
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let info = FstInfo {
        start_time: 0,
        timescale_exponent: -9, // 1ns
        version: "fst-reader test".to_string(),
        date: String::new(),
        file_type: FstFileType::Verilog,
    };

    let mut header = open_fst(path, &info).unwrap();

    // Add a scope with two signals
    header.scope("top", "", FstScopeType::Module).unwrap();
    let clk = header
        .var(
            "clk",
            FstSignalType::bit_vec(1),
            FstVarType::Wire,
            FstVarDirection::Implicit,
            None,
        )
        .unwrap();
    let counter = header
        .var(
            "counter",
            FstSignalType::bit_vec(8),
            FstVarType::Reg,
            FstVarDirection::Implicit,
            None,
        )
        .unwrap();
    header.up_scope().unwrap();

    // Write some value changes
    let mut body = header.finish().unwrap();

    // Initial values at time 0
    body.time_change(0).unwrap();
    body.signal_change(clk, b"0").unwrap();
    body.signal_change(counter, b"00000000").unwrap();

    // Some transitions
    body.time_change(10).unwrap();
    body.signal_change(clk, b"1").unwrap();

    body.time_change(20).unwrap();
    body.signal_change(clk, b"0").unwrap();
    body.signal_change(counter, b"00000001").unwrap();

    body.time_change(30).unwrap();
    body.signal_change(clk, b"1").unwrap();

    body.time_change(40).unwrap();
    body.signal_change(clk, b"0").unwrap();
    body.signal_change(counter, b"00000010").unwrap();

    body.finish().unwrap();

    temp_file
}

/// Test that a file written by fst-writer can be read by fst-reader.
///
/// This tests the basic roundtrip: write with fst-writer, read with fst-reader.
#[test]
fn test_writer_reader_roundtrip_simple() {
    let temp_file = create_simple_fst();

    // Read it back with fst-reader
    let file = std::fs::File::open(temp_file.path()).unwrap();
    let mut reader = FstReader::open(BufReader::new(file)).unwrap();

    // Read hierarchy and verify
    let mut var_count = 0;
    let mut scope_count = 0;
    reader
        .read_hierarchy(|entry| match entry {
            FstHierarchyEntry::Var { name, length, .. } => {
                var_count += 1;
                // Verify we got the expected signals
                assert!(
                    name == "clk" || name == "counter",
                    "Unexpected variable: {}",
                    name
                );
                if name == "clk" {
                    assert_eq!(length, 1);
                } else {
                    assert_eq!(length, 8);
                }
            }
            FstHierarchyEntry::Scope { name, .. } => {
                scope_count += 1;
                assert_eq!(name, "top");
            }
            _ => {}
        })
        .unwrap();

    assert_eq!(var_count, 2, "Should have found 2 variables");
    assert_eq!(scope_count, 1, "Should have found 1 scope");
}

/// Test that a file written by fst-writer can be re-read multiple times.
///
/// This verifies that the file format is self-consistent and doesn't have
/// any position-dependent bugs.
#[test]
fn test_refilter_fst_writer_output() {
    let temp_file = create_simple_fst();

    // Read it multiple times to ensure consistency
    for i in 0..3 {
        let file = std::fs::File::open(temp_file.path()).unwrap();
        let result = FstReader::open(BufReader::new(file));

        match result {
            Ok(mut reader) => {
                let mut signals = Vec::new();
                reader
                    .read_hierarchy(|entry| {
                        if let FstHierarchyEntry::Var { handle, .. } = entry {
                            signals.push(handle);
                        }
                    })
                    .unwrap();

                assert_eq!(
                    signals.len(),
                    2,
                    "Iteration {}: Should have found 2 signals",
                    i
                );
            }
            Err(e) => {
                panic!(
                    "Iteration {}: Failed to read fst-writer output: {:?}. \
                     This likely indicates the section_length bug.",
                    i, e
                );
            }
        }
    }
}
