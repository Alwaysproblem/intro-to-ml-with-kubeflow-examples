# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Lightweight python components
# 
# Lightweight python components do not require you to build a new container image for every code change.
# They're intended to use for fast iteration in notebook environment.
# 
# #### Building a lightweight python component
# To build a component just define a stand-alone python function and then call kfp.components.func_to_container_op(func) to convert it to a component that can be used in a pipeline.
# 
# There are several requirements for the function:
# * The function should be stand-alone. It should not use any code declared outside of the function definition. Any imports should be added inside the main function. Any helper functions should also be defined inside the main function.
# * The function can only import packages that are available in the base image. If you need to import a package that's not available you can try to find a container image that already includes the required packages. (As a workaround you can use the module subprocess to run pip install for the required package. There is an example below in my_divmod function.)
# * If the function operates on numbers, the parameters need to have type hints. Supported types are ```[int, float, bool]```. Everything else is passed as string.
# * To build a component with multiple output values, use the typing.NamedTuple type hint syntax: ```NamedTuple('MyFunctionOutputs', [('output_name_1', type), ('output_name_2', float)])```

# %%
# Install the SDK
#!pip3 install 'kfp>=0.1.31.2' --quiet


# %%
import kfp
import kfp.components as comp

# %% [markdown]
# Simple function that just add two numbers:

# %%
#Define a Python function
def add(a: float, b: float) -> float:
   '''Calculates sum of two arguments'''
   return a + b

# %% [markdown]
# Convert the function to a pipeline operation

# %%
add_op = comp.func_to_container_op(add)

# %% [markdown]
# A bit more advanced function which demonstrates how to use imports, helper functions and produce multiple outputs.
# %% [markdown]
# `mlpipeline_ui_metadata`, `UI_metadata` is the key to pipeline UI metadata and artifacts store.
# 
# There are 2 possible way to show the visualization.
# 
# - Write your own components refer to [confusion matrix components](https://github.com/kubeflow/pipelines/blob/master/components/local/confusion_matrix/component.yaml) which is the same with offical tutorial. This need to write json message to `/mlpipeline-ui-metadata.json` and `/mlpipeline-metrics.json` file. The path of those file can be costomized. The base docker image can be costomized to meet your own demonds.
# 
# - Return from the function or `Container Oprations` like example below.
# 

# %%

#Advanced function
#Demonstrates imports, helper functions and multiple outputs
from typing import NamedTuple
def my_divmod(dividend: float, divisor:float) -> NamedTuple('MyDivmodOutput', [('quotient', float), ('remainder', float), ('mlpipeline_ui_metadata', 'UI_metadata'), ('mlpipeline_metrics', 'Metrics')]):
    '''Divides two numbers and calculate  the quotient and remainder'''
    
    #Imports inside a component function:
    import numpy as np

    #This function demonstrates how to use nested functions inside a component function:
    def divmod_helper(dividend, divisor):
        return np.divmod(dividend, divisor)

    (quotient, remainder) = divmod_helper(dividend, divisor)

    from tensorflow.python.lib.io import file_io
    import json
    
    # Exports a sample tensorboard:
    metadata = {
      'outputs' : [{
        'type': 'tensorboard',
        'source': 'gs://ml-pipeline-dataset/tensorboard-train',
      }]
    }

    # Exports two sample metrics:
    metrics = {
      'metrics': [{
          'name': 'quotient',
          'numberValue':  float(quotient),
        },{
          'name': 'remainder',
          'numberValue':  float(remainder),
        }]}

    from collections import namedtuple
    divmod_output = namedtuple('MyDivmodOutput', ['quotient', 'remainder', 'mlpipeline_ui_metadata', 'mlpipeline_metrics'])
    # note : `metadata` `metrics` need to return with json format.
    # note : need to return a namedtuple with keywords `mlpipeline_ui_metadata` `mlpipeline_metrics`
    return divmod_output(quotient, remainder, json.dumps(metadata), json.dumps(metrics))

# %% [markdown]
# Test running the python function directly

# %%
my_divmod(100, 7)

# %% [markdown]
# #### Convert the function to a pipeline operation
# 
# You can specify an alternative base container image (the image needs to have Python 3.5+ installed).

# %%
divmod_op = comp.func_to_container_op(my_divmod, base_image='tensorflow/tensorflow:1.11.0-py3')

# %% [markdown]
# #### Define the pipeline
# Pipeline function has to be decorated with the `@dsl.pipeline` decorator

# %%
import kfp.dsl as dsl
@dsl.pipeline(
   name='Calculation pipeline',
   description='A toy pipeline that performs arithmetic calculations.'
)
def calc_pipeline(
   a='a',
   b='7',
   c='17',
):
    #Passing pipeline parameter and a constant value as operation arguments
    add_task = add_op(a, 4) #Returns a dsl.ContainerOp class instance. 
    
    #Passing a task output reference as operation arguments
    #For an operation with a single return value, the output reference can be accessed using `task.output` or `task.outputs['output_name']` syntax
    divmod_task = divmod_op(add_task.output, b)

    #For an operation with a multiple return values, the output references can be accessed using `task.outputs['output_name']` syntax
    result_task = add_op(divmod_task.outputs['quotient'], c)

# %% [markdown]
# #### Submit the pipeline for execution

# %%
#Specify pipeline argument values
arguments = {'a': '7', 'b': '8'}

#Submit a pipeline run
kfp.Client().create_run_from_pipeline_func(calc_pipeline, arguments=arguments)

# Run the pipeline on a separate Kubeflow Cluster instead
# (use if your notebook is not running in Kubeflow - e.x. if using AI Platform Notebooks)
# kfp.Client(host='<ADD KFP ENDPOINT HERE>').create_run_from_pipeline_func(calc_pipeline, arguments=arguments)

#vvvvvvvvv This link leads to the run information page. (Note: There is a bug in JupyterLab that modifies the URL and makes the link stop working)


# %%
cli = kfp.Client()


# %%
cli.experiments.list_experiment()


# %%
cli.experiments.delete_experiment('ce36fa2b-c279-43fb-8b72-2bffeb922b5f')


# %%
cli.runs.delete_run('17adcc3c-e6d2-45b8-a3cf-5cfa3fe4c8df')


# %%



