use anyhow::Result;
use tch::nn::{FuncT, Linear, Module, ModuleT, OptimizerConfig};
use tch::vision::{self, imagenet, resnet};
use tch::{nn, Device, Kind};

fn deploy() -> Result<(), Box<dyn std::error::Error>> {
    let device = tch::Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // Load the ResNet18 model without the final layer
    let resnet18 = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load("./weights/resnet18.ot")?;

    // Load the trained linear layer
    let mut linear_vs = nn::VarStore::new(device);
    let linear = nn::linear(linear_vs.root(), 512, 2, Default::default());
    linear_vs.load("./weights/resnet18_linear.ot")?;

    // Process image
    let image_path = "./dataset/train/bird/chicken.jpeg";
    let image = imagenet::load_image_and_resize224(image_path)?;
    let image = image.unsqueeze(0); // Add batch dimension
    let image = image.to_device(device); // Ensure the image is on the same device as the model

    // Apply model
    let resnet_features = tch::no_grad(|| resnet18.forward_t(&image, false));
    let logits = tch::no_grad(|| linear.forward(&resnet_features));

    let output = logits.softmax(-1, Kind::Float);

    let (top_probs, top_idxs) = output.topk(2, -1, true, true);

    let labels = vec!["drone", "bird"];

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

fn train() -> Result<()> {
    // Download images of birds and drones from a kaggle dataset
    // https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset

    // Set up GPU
    let device = tch::Device::cuda_if_available();

    // Load the pretrianed ResNet18 model
    let mut vs = nn::VarStore::new(device);
    let net = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load(std::path::Path::new("./weights/resnet18.ot"))
        .unwrap();

    // Pre-compute the final activations.
    let dataset = imagenet::load_from_dir(std::path::Path::new("./dataset")).unwrap();
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    for epoch_idx in 1..1001 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        sgd.backward_step(&loss);

        let test_accuracy = test_images
            .apply(&linear)
            .accuracy_for_logits(&dataset.test_labels);
        println!("{} {:.2}%", epoch_idx, 100. * f64::try_from(test_accuracy)?);
    }

    // Save the model
    vs.save(std::path::Path::new("./weights/resnet18_linear.ot"))?;
    println!("Saved weights to ./weights/resnet18_linear.ot");

    Ok(())
}

fn load_resnet_model() -> Result<FuncT<'static>> {
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load(std::path::Path::new("./weights/resnet18.ot"))?;

    Ok(net)
}

fn load_linear_model() -> Result<Linear> {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let linear = nn::linear(vs.root(), 512, 2, Default::default());
    vs.load("./weights/resnet18_linear.ot")?;

    println!("Loaded weights from ./weights/resnet18_linear.ot");
    Ok(linear)
}

fn load_pre_trianed_model() -> Result<()> {
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image =
        imagenet::load_image_and_resize224("dataset/train/bird/singleBirdinsky0.jpeg").unwrap();

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = Box::new(resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT));
    vs.load("./weights/resnet18.ot")?;

    // Apply the forward pass of the model to get the logits.
    let output = net
        .forward_t(&image.unsqueeze(0), /* train= */ false)
        .softmax(-1, tch::Kind::Float); // Convert to probability.

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    deploy()?;

    Ok(())
}
