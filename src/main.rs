use anyhow::Result;
use tch::nn::OptimizerConfig;
use tch::vision::imagenet;
use tch::{nn, vision};

fn main() -> Result<()> {
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
