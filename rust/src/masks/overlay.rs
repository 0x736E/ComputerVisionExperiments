use opencv::core::Mat;
use opencv::Error;

pub trait OverlayProcessor <'a> {
    fn draw(
        &mut self,
        frame: &Mat,
    ) -> Result<Mat, Error>;
}