//! Robust, normalized homography fitting from all marker corners.
use anyhow::{Context, Result, bail};
use nalgebra::{DMatrix, Matrix3};
use opencv::{core::Mat, prelude::*};

pub type Point = (f64, f64);

pub fn project(h: &Mat, (x, y): Point) -> Result<Point> {
    let v = |r, c| h.at_2d::<f64>(r, c).copied();
    let w = v(2, 0)? * x + v(2, 1)? * y + v(2, 2)?;
    if !w.is_finite() || w.abs() < 1e-12 {
        bail!("Homography projects a point to infinity");
    }
    Ok((
        (v(0, 0)? * x + v(0, 1)? * y + v(0, 2)?) / w,
        (v(1, 0)? * x + v(1, 1)? * y + v(1, 2)?) / w,
    ))
}

pub fn corner_order(
    corners: &[Point; 4],
    expected: &[Point; 4],
    initial: &Mat,
) -> Result<[Point; 4]> {
    let mut mapped = [(0.0, 0.0); 4];
    for (output, &input) in mapped.iter_mut().zip(corners) {
        if !input.0.is_finite() || !input.1.is_finite() {
            bail!("Non-finite marker corner");
        }
        *output = project(initial, input)?;
    }
    // Decoder corner start/winding and photograph rotation are independent.
    // Match bijectively in paper coordinates, never by raw image X/Y ordering.
    let mut best = (f64::INFINITY, [0, 1, 2, 3]);
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let order = [a, b, c, d];
                    if (0..4).any(|i| (i + 1..4).any(|j| order[i] == order[j])) {
                        continue;
                    }
                    let cost: f64 = (0..4)
                        .map(|i| {
                            let p = mapped[order[i]];
                            (p.0 - expected[i].0).powi(2) + (p.1 - expected[i].1).powi(2)
                        })
                        .sum();
                    if cost < best.0 {
                        best = (cost, order);
                    }
                }
            }
        }
    }
    Ok(best.1.map(|i| corners[i]))
}

fn normalize(points: &[Point]) -> Result<(Matrix3<f64>, Vec<Point>)> {
    let n = points.len() as f64;
    let center = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let distance = points
        .iter()
        .map(|p| (p.0 - center.0).hypot(p.1 - center.1))
        .sum::<f64>()
        / n;
    if !distance.is_finite() || distance < 1e-9 {
        bail!("Degenerate calibration points");
    }
    let s = 2.0_f64.sqrt() / distance;
    let t = Matrix3::new(s, 0.0, -s * center.0, 0.0, s, -s * center.1, 0.0, 0.0, 1.0);
    Ok((
        t,
        points
            .iter()
            .map(|p| ((p.0 - center.0) * s, (p.1 - center.1) * s))
            .collect(),
    ))
}

fn weighted_fit(source: &[Point], target: &[Point], weights: &[f64]) -> Result<Mat> {
    let (ts, source) = normalize(source)?;
    let (tt, target) = normalize(target)?;
    let mut a = DMatrix::<f64>::zeros(source.len() * 2, 9);
    for (i, (&(x, y), &(u, v))) in source.iter().zip(&target).enumerate() {
        let w = weights[i].sqrt();
        for (j, value) in [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
            .iter()
            .enumerate()
        {
            a[(2 * i, j)] = value * w;
        }
        for (j, value) in [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
            .iter()
            .enumerate()
        {
            a[(2 * i + 1, j)] = value * w;
        }
    }
    let svd = a.svd(false, true);
    if svd.singular_values[7] < 1e-9 {
        bail!("Degenerate marker corner layout");
    }
    let vt = svd.v_t.context("Missing calibration SVD")?;
    let row: Vec<f64> = (0..9).map(|i| vt[(8, i)]).collect();
    let h = tt.try_inverse().context("Invalid target normalization")?
        * Matrix3::from_row_slice(&row)
        * ts;
    if !h.iter().all(|x| x.is_finite()) || h[(2, 2)].abs() < 1e-12 {
        bail!("Invalid fitted homography");
    }
    let h = h / h[(2, 2)];
    // Reject a singular solution before passing it to warpPerspective.
    if h.try_inverse().is_none() {
        bail!("Singular fitted homography");
    }
    Ok(Mat::from_slice_2d(&[
        [h[(0, 0)], h[(0, 1)], h[(0, 2)]],
        [h[(1, 0)], h[(1, 1)], h[(1, 2)]],
        [h[(2, 0)], h[(2, 1)], h[(2, 2)]],
    ])?)
}

pub struct Fit {
    pub homography: Mat,
    pub residuals_px: Vec<f64>,
    pub downweighted: usize,
}

pub fn fit_corners(
    source: &[Point],
    target: &[Point],
    initial: &Mat,
    huber_px: f64,
) -> Result<Fit> {
    if source.len() < 5 || source.len() != target.len() || !huber_px.is_finite() || huber_px <= 0.0
    {
        bail!("Invalid corner correspondences");
    }
    let mut h = initial.try_clone()?;
    let residuals = |h: &Mat| -> Result<Vec<f64>> {
        source
            .iter()
            .zip(target)
            .map(|(&a, &b)| {
                let p = project(h, a)?;
                Ok((p.0 - b.0).hypot(p.1 - b.1))
            })
            .collect()
    };
    // Huber IRLS keeps all 16 observations, with bounded influence for bad
    // corners. Residuals and the threshold are in rectified paper pixels.
    for _ in 0..20 {
        let errors = residuals(&h)?;
        let weights: Vec<f64> = errors
            .iter()
            .map(|e| if *e <= huber_px { 1.0 } else { huber_px / e })
            .collect();
        let next = weighted_fit(source, target, &weights)?;
        let change = source
            .iter()
            .map(|&p| -> Result<f64> {
                let (a, b) = (project(&h, p)?, project(&next, p)?);
                Ok((a.0 - b.0).hypot(a.1 - b.1))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        h = next;
        if change < 1e-5 {
            break;
        }
    }
    let errors = residuals(&h)?;
    let downweighted = errors.iter().filter(|&&e| e > huber_px).count();
    Ok(Fit {
        homography: h,
        residuals_px: errors,
        downweighted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Mat {
        Mat::from_slice_2d(&[[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap()
    }

    #[test]
    fn a_bad_corner_is_downweighted_instead_of_dragging_the_whole_page() {
        let mut target = Vec::new();
        for (x, y) in [
            (175.0, 175.0),
            (1830.0, 175.0),
            (1830.0, 2850.0),
            (175.0, 2850.0),
        ] {
            target.extend([
                (x, y),
                (x + 470.0, y),
                (x + 470.0, y + 470.0),
                (x, y + 470.0),
            ]);
        }
        let mut source = target.clone();
        source[0].0 += 100.0;
        source[0].1 += 80.0;
        let robust = fit_corners(&source, &target, &identity(), 4.0).unwrap();
        let ordinary = weighted_fit(&source, &target, &vec![1.0; 16]).unwrap();
        let evaluate = |h: &Mat| {
            let p = project(h, (1100.0, 1700.0)).unwrap();
            (p.0 - 1100.0).hypot(p.1 - 1700.0)
        };
        assert!(robust.downweighted >= 1);
        assert!(robust.residuals_px[0] > 100.0);
        assert!(evaluate(&robust.homography) < 2.0);
        assert!(evaluate(&ordinary) > 3.0 * evaluate(&robust.homography));
    }

    #[test]
    fn corner_matching_handles_decoder_start_and_winding() {
        let expected = [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)];
        for offset in 0..4 {
            let corners = std::array::from_fn(|i| expected[(offset + 4 - i) % 4]);
            assert_eq!(
                corner_order(&corners, &expected, &identity()).unwrap(),
                expected
            );
        }
        let degenerate = vec![(1.0, 2.0); 16];
        assert!(fit_corners(&degenerate, &degenerate, &identity(), 4.0).is_err());
    }
}
