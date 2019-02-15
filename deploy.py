import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.model import Model
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice

auth_config = InteractiveLoginAuthentication(False, "72f988bf-86f1-41af-91ab-2d7cd011db47")
ws=Workspace.from_config('aml_config/config.json', auth_config)
ws.get_details()

myenv = CondaDependencies()
myenv.add_conda_package("numpy")
myenv.add_conda_package("keras")
myenv.add_conda_package("tensorflow")
myenv.add_conda_package("PIL")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())

# Register a trained model
print('Registering model...')
model = Model.register(model_path = "modelfiles",
                       model_name = "dogs-vs-cat",
                       description = "ready lab 314",
                       workspace = ws)

# Image configuration
print('Creating image configuration...')
image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                 runtime = "python",
                                                 conda_file = "myenv.yml",
                                                 description = "Dogs vs Cats classification model",
                                                 )

# Register the image from the image configuration in ACR
print('Creating and registering image with workspace...')
image = ContainerImage.create(name = "dogscats", 
                              models = [model],
                              image_config = image_config,
                              workspace = ws
                              )

image.wait_for_creation(show_output=True)
print(image.image_build_log_uri)

# Deploy to ACI
print('Creating ACI deployment configuration...')
aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 2, 
                                               tags = {"data": "dogscatsmodel", "type": "classification"}, 
                                               description = 'Dogs vs Cats classification service')


print('Starting web service deployment...')
service_name = 'dogscats-1502-1'
service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                            image = image,
                                            name = service_name,
                                            workspace = ws)
service.wait_for_deployment(show_output = True)

print(service.state)
print('Fetching the scoring endpoint URI...')
print(service.scoring_uri)
