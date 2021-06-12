# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from abc import ABC
import io
import base64
import torch
from PIL import Image
from captum.attr import IntegratedGradients


"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import importlib.util
import time
import torch


"""
Util files for TorchServe
"""
import inspect
import os
import json
import itertools
import logging

logger = logging.getLogger(__name__)

def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.
    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                    member.__module__ == module.__name__)]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes

def load_label_mapping(mapping_file_path):
    """
    Load a JSON mapping { class ID -> friendly class name }.
    Used in BaseHandler.
    """
    if not os.path.isfile(mapping_file_path):
        logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
        return None

    with open(mapping_file_path) as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise Exception('index_to_name mapping should be in "class":"label" json format')

    # Older examples had a different syntax than others. This code accommodates those.
    if 'object_type_names' in mapping and isinstance(mapping['object_type_names'], list):
        mapping = {str(k): v for k, v in enumerate(mapping['object_type_names'])}
        return mapping

    for key, value in mapping.items():
        new_value = value
        if isinstance(new_value, list):
            new_value = value[-1]
        if not isinstance(new_value, str):
            raise Exception('labels in index_to_name must be either str or [str]')
        mapping[key] = new_value
    return mapping

def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not (isinstance(probs, list) and isinstance(probs, list)):
        raise Exception('Convert classes to list before doing mapping')
    if mapping is not None and not isinstance(mapping, dict):
        raise Exception('Mapping must be a dict')

    if lbl_classes is None:
        lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

    results = [
        {
            (mapping[str(lbl_class)] if mapping is not None else str(lbl_class)): prob
            for lbl_class, prob in zip(*row)
        }
        for row in zip(lbl_classes, probs)
    ]

    return results


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
	   First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        self.model.eval()

        logger.debug('Model file %s loaded successfully', model_pt_path)

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.
        Args:
            model_pt_path (str): denotes the path of the model file.
        Returns:
            (NN Model Object) : Loads the model object.
        """
        return torch.jit.load(model_pt_path, map_location=self.device)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.
        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.
        Raises:
            RuntimeError: It raises this error when the model.py file is missing.
            ValueError: Raises value error when there is more than one class in the label,
                        since the mapping supports only one label per class.
        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class()
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        return model

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        return torch.as_tensor(data, device=self.device)

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)
        return results

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return data.tolist()

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            output = self.inference(data_preprocess)
            output = self.postprocess(output)
        else :
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output

    def explain_handle(self, data_preprocess, raw_data):
        """Captum explanations handler
        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request
        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = None
        inputs = None
        target = 0

        logger.info("Calculating Explanations")
        row = raw_data[0]
        if isinstance(row, dict):
            logger.info("Getting data and target")
            inputs = row.get("data") or row.get("body")
            target = row.get("target")
            if not target:
                target = 0

        output_explain = self.get_insights(data_preprocess, inputs, target)
        return output_explain

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                self.explain = True
                return True
        return False
    
    

class VisionHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()