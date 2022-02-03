use serde_yaml;

use std::env;
use std::path::PathBuf;
use std::collections::{HashMap,hash_map::Entry};

use std::io::prelude::*;

use std::fmt::format;

fn create_line(name: String, coeffs: &[f64;4]) -> String {
    format!("    pub const {}: [f64;4] = [ {}, {}, {}, {} ];\n",
        name,
        coeffs[0].to_string(),
        coeffs[1].to_string(),
        coeffs[2].to_string(),
        coeffs[3].to_string()
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load yaml data
    let yaml_file = std::fs::File::open("c_m_delta_elev_fits.yaml")?;
    let coeffs: HashMap<String,[f64;4]> = serde_yaml::from_reader(yaml_file)?;
    
    let target_keys: Vec<(String,String)> = vec![
        ("c_m_delta_elev@0.2/10.0/-10".to_string(),"THR_0_2_ASPD_10_0".to_string())
    ];
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("c_m_delta_elev_fits.rs");
    
    let mut constants_file = std::fs::File::create(out_path)?;
    
    constants_file.write_all("mod c_m_delta_elev_coeffs {\n".as_bytes());
    
    for (key,name) in target_keys {
        if let Some(entry) = coeffs.get(&key) {
            let line = create_line(name,entry);
            println!("{}",line);
            constants_file.write_all(line.as_bytes());
        }
    }
    
    constants_file.write_all("}".as_bytes());

    Ok(())
}