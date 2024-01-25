<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Kosmos-2 Module

This repository contains the code supporting the Kosmos-2 base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Kosmos-2](https://github.com/microsoft/unilm/tree/master/kosmos-2), developed by Microsoft, is a multimodal language model that you can use for zero-shot object detection. You can use Kosmos-2 with autodistill for object detection.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Kosmos-2 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/kosmos_2/).

## Installation

To use Kosmos-2 with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-kosmos-2
```

## Quickstart

```python
from autodistill_kosmos_2 import Kosmos2

# define an ontology to map class names to our Kosmos2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Kosmos2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

predictions = base_model.predict("./example.png")

base_model.label("./context_images", extension=".jpeg")
```


## License

This package is implemented using the [Transformers Kosmos-2 implementation](https://huggingface.co/microsoft/kosmos-2-patch14-224). The underlying Kosmos-2 model, developed by Microsoft, is licensed under an [MIT license](https://github.com/microsoft/unilm/blob/master/LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
