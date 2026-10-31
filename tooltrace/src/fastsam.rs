//! Local model discovery and reversible FastSAM image preprocessing.
use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use opencv::{
    core::{Mat, Rect, Size},
    imgproc,
    prelude::*,
};
use std::path::{Path, PathBuf};

pub fn find_model(explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    let configured = explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("TOOLTRACE_FASTSAM_MODEL").map(PathBuf::from));
    let executable = std::env::current_exe().context("Could not locate tooltrace executable")?;
    let cwd = std::env::current_dir()?;
    resolve_model(configured.as_deref(), &executable, &cwd)
}

fn resolve_model(
    configured: Option<&Path>,
    executable: &Path,
    cwd: &Path,
) -> Result<Option<PathBuf>> {
    if let Some(path) = configured {
        if !path.is_file() {
            bail!(
                "Configured FastSAM ONNX model does not exist: {}",
                path.display()
            );
        }
        return Ok(Some(path.canonicalize()?));
    }
    // Works from target/debug, the repository root, or an unrelated working
    // directory. A packaged executable can carry its own models/ directory.
    for root in executable
        .parent()
        .into_iter()
        .flat_map(Path::ancestors)
        .chain(cwd.ancestors())
    {
        for name in ["FastSAM-x.onnx", "FastSAM-s.onnx", "FastSAM.onnx"] {
            let candidate = root.join("models").join(name);
            if candidate.is_file() {
                return Ok(Some(candidate.canonicalize()?));
            }
        }
    }
    let legacy = Path::new("D:/data/FastSam-S.onnx");
    Ok(legacy.is_file().then(|| legacy.to_path_buf()))
}

#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub size: u32,
    pub width: u32,
    pub height: u32,
    pub left: u32,
    pub top: u32,
    original_width: u32,
    original_height: u32,
}

impl Letterbox {
    pub fn new(width: u32, height: u32, size: u32) -> Self {
        let scale = size as f64 / width.max(height) as f64;
        let resized_width = (width as f64 * scale).round().clamp(1.0, size as f64) as u32;
        let resized_height = (height as f64 * scale).round().clamp(1.0, size as f64) as u32;
        Self {
            size,
            width: resized_width,
            height: resized_height,
            left: (size - resized_width) / 2,
            top: (size - resized_height) / 2,
            original_width: width,
            original_height: height,
        }
    }

    pub fn prepare(self, image: &RgbImage) -> RgbImage {
        let resized = image::imageops::resize(
            image,
            self.width,
            self.height,
            image::imageops::FilterType::Triangle,
        );
        let mut padded = RgbImage::from_pixel(self.size, self.size, Rgb([114, 114, 114]));
        image::imageops::replace(&mut padded, &resized, self.left as i64, self.top as i64);
        padded
    }

    pub fn point_to_original(self, x: f32, y: f32) -> (f32, f32) {
        (
            (x - self.left as f32) * self.original_width as f32 / self.width as f32,
            (y - self.top as f32) * self.original_height as f32 / self.height as f32,
        )
    }

    pub fn size_to_original(self, w: f32, h: f32) -> (f32, f32) {
        (
            w * self.original_width as f32 / self.width as f32,
            h * self.original_height as f32 / self.height as f32,
        )
    }

    pub fn restore_mask(self, prototype: &impl opencv::core::ToInputArray) -> Result<Mat> {
        let mut square = Mat::default();
        imgproc::resize(
            prototype,
            &mut square,
            Size::new(self.size as i32, self.size as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        let unpadded = Mat::roi(
            &square,
            Rect::new(
                self.left as i32,
                self.top as i32,
                self.width as i32,
                self.height as i32,
            ),
        )?;
        let mut restored = Mat::default();
        imgproc::resize(
            &unpadded,
            &mut restored,
            Size::new(self.original_width as i32, self.original_height as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portrait_coordinates_and_masks_remove_letterbox_padding() {
        let l = Letterbox::new(2481, 3508, 1024);
        let p = l.point_to_original(l.left as f32, l.top as f32);
        assert_eq!(p, (0.0, 0.0));
        let p = l.point_to_original((l.left + l.width) as f32, (l.top + l.height) as f32);
        assert_eq!(p, (2481.0, 3508.0));
        let mut square = Mat::new_rows_cols_with_default(
            1024,
            1024,
            opencv::core::CV_32F,
            opencv::core::Scalar::all(0.0),
        )
        .unwrap();
        imgproc::rectangle(
            &mut square,
            Rect::new(l.left as i32, l.top as i32, l.width as i32, l.height as i32),
            opencv::core::Scalar::all(1.0),
            -1,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
        let restored = l.restore_mask(&square).unwrap();
        assert_eq!(restored.size().unwrap(), Size::new(2481, 3508));
        assert!(restored
            .data_typed::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v > 0.99));
    }

    #[test]
    fn model_is_found_relative_to_executable_and_missing_override_is_an_error() {
        let unique = format!(
            "tooltrace-model-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(root.join("models")).unwrap();
        let model = root.join("models/FastSAM-x.onnx");
        std::fs::write(&model, b"discovery fixture").unwrap();
        let result = resolve_model(
            None,
            &root.join("target/debug/tooltrace.exe"),
            &root.join("unrelated"),
        )
        .unwrap();
        assert_eq!(result.unwrap(), model.canonicalize().unwrap());
        assert!(resolve_model(
            Some(&root.join("missing.onnx")),
            &root.join("tooltrace.exe"),
            &root
        )
        .is_err());
        std::fs::remove_file(model).unwrap();
        std::fs::remove_dir(root.join("models")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
