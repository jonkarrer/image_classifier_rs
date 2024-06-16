use anyhow::Result;
use tch::{
    nn::{self, FuncT, Linear, Module, ModuleT},
    vision::{self, imagenet},
    Device, Kind, Tensor,
};

fn load_resnet_no_final_layer(weights_path: &str, device: Device) -> FuncT<'static> {
    let mut vs = nn::VarStore::new(device);
    let net = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load(std::path::Path::new(weights_path))
        .expect("Failed to load resnet weights");

    println!("Loaded resnet18 model from {}", weights_path);
    net
}

fn load_trained_layer(weights_path: &str, device: Device) -> Linear {
    let mut vs = nn::VarStore::new(device);
    let linear = nn::linear(vs.root(), 512, 2, Default::default());
    vs.load(weights_path)
        .expect("Failed to load trained weights");

    println!("Loaded linear model from {}", weights_path);
    linear
}

fn process_test_image(image_path: &str, device: Device) -> Tensor {
    imagenet::load_image_and_resize224(image_path)
        .expect("Failed to load image")
        .unsqueeze(0) // Add batch dimension
        .to_device(device) // Make sure it's on the GPU
}

pub fn run_test_on_image(image_path: &str) -> Result<()> {
    let device = tch::Device::cuda_if_available();
    let test_image = process_test_image(image_path, device);

    // Pass image through the base resnet model
    let resnet_features = tch::no_grad(|| {
        load_resnet_no_final_layer("weights/resnet18.ot", device).forward_t(&test_image, false)
    });

    // Pass the resnet features through the linear model
    let logits = tch::no_grad(|| {
        load_trained_layer("weights/resnet18_linear.ot", device).forward(&resnet_features)
    });

    // Get the top 2 predictions
    let labels = vec!["drone", "bird"];
    let output = logits.softmax(-1, Kind::Float);
    let (top_probs, top_idxs) = output.topk(2, -1, true, true);

    println!("I think..:");
    for i in 0..2 {
        let prob = top_probs.double_value(&[0, i]);
        let idx = top_idxs.int64_value(&[0, i]) as usize;
        if let Some(class_name) = labels.get(idx) {
            println!("{:50} {:5.2}%", class_name, 100.0 * prob);
        }
    }

    Ok(())
}
