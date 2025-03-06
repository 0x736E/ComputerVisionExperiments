use opencv::core::{add_weighted, Mat, MatExprTraitConst, MatTraitConst, Rect, Scalar, CV_8UC3};
use opencv::Error;
use opencv::imgproc::{resize, INTER_LINEAR};
use crate::detectors::motion_mog2::MotionMog2;
use crate::masks::overlay::OverlayProcessor;
use opencv::prelude::MatTrait;

const MAX_BOUNDING_BOXES: usize = 10;

pub struct MotionOverlay<'a> {
    mog2_detector: &'a MotionMog2,
    tint_color: Scalar,
    bounding_boxes: Vec<Rect>,
}

impl<'a> MotionOverlay<'a> {
    pub fn new(motion_mog2: &'a MotionMog2) -> Self {
        Self {
            mog2_detector: &motion_mog2,
            tint_color: Scalar::new(
                0.0,
                0.0,
                255.0,
                0.0),
            bounding_boxes: Vec::with_capacity(MAX_BOUNDING_BOXES)
        }
    }

    pub fn get_bounding_boxes(&self) -> &Vec<Rect> {
        &self.bounding_boxes
    }
}

fn tint_mask(mask: &Mat, color: &Scalar) -> Result<Mat, Error> {

    let mut tinted_mask = Mat::zeros(
        mask.rows(),
        mask.cols(),
        CV_8UC3
    )?.to_mat()?;

    tinted_mask.set_to(
        &color,
        &mask
    )?;

    Ok(tinted_mask.clone())
}

impl<'a> OverlayProcessor<'a> for MotionOverlay<'a> {
    fn draw(&mut self, frame: &Mat) -> Result<Mat, Error> {

        let mut overlay = frame.clone();
        let mut tinted_mask = tint_mask(&self.mog2_detector.get_diff_mask(), &self.tint_color)?;

        resize(
            &tinted_mask,
            &mut overlay,
            frame.size()?,
            0.0,
            0.0,
            INTER_LINEAR
        )?;

        add_weighted(
            &frame,
            1.0,
            &overlay,
            0.5,
            0.0,
            &mut tinted_mask,
            -1,
        )?;

        for current_rect in self.mog2_detector.get_bounding_boxes().iter() {
            if !self.bounding_boxes.iter().any(|outer_rect: &Rect| {
                outer_rect.contains(current_rect.tl()) && outer_rect.contains(current_rect.br())
            }) {
                opencv::imgproc::rectangle(
                    &mut tinted_mask,
                    Rect::new(
                        current_rect.x * 2,
                        current_rect.y * 2,
                        current_rect.width * 2,
                        current_rect.height * 2,
                    ),
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )?;
            }
        }

        Ok(tinted_mask.clone())
    }
}