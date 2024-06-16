pub mod train_model;
pub mod use_trained_model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use_trained_model::run_test_on_image("dataset/val/bird/singleBirdinsky350.jpeg")?;

    Ok(())
}
