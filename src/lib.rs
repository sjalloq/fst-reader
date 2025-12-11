// Copyright 2023 The Regents of the University of California
// Copyright 2024 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

mod fastlz;
mod io;
mod raw_reader;
mod reader;
mod types;

// Re-export core reading types
pub use io::ReaderError;
pub use reader::{FstFilter, FstHeader, FstReader, FstSignalValue, is_fst_file};
pub use types::{
    FstHierarchyEntry, FstScopeType, FstSignalHandle, FstVarDirection, FstVarType, FstVhdlDataType,
    FstVhdlVarType,
};

// Re-export raw access types for filtered copying
pub use raw_reader::{
    PositionTable, SignalDataLocation, VcBlockInfo, VcBlockKind, VcBlockReader, VcPackType,
};
