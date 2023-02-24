# domain_classification
A binary domain classifier based on positive labels and an active learning algorithm.

For more information, go to the [documentation pages](https://intelcomph2020.github.io/domain_classification/index.html).


# Dockerfile explanation
The dockerfile contains all the instructions to generate a docker image, a template to build a docker container which contains the application.

- Base image: the base image for the application is ubuntu+python
- Directories architecture: the application will be in the */app* directory inside the container, and the */shared_data* directory will contain shared data with host.
- Application: copy application info to current directory (*/app*)
- Execution: the command in **CMD** is the default option when executing the container and can be overwritten by the user.


## Create Docker image with WordList manager
Use the following command to create a docker image.

The flag *-t* specifies the name of the image that will be created with an optional tag (for example its version).
```
docker build <-t NAME:tag> <Dockerfile location>
```
### Example:
```
docker build -t dom-class .
```
- The name of the image in this case is *dom-class*, with no specific version.
- The location of the `Dockerfile` is the current directory.

## Run image container
To run a container with the previous configuration, the following command is needed:
```
docker run <--rm> -i <--name CONTAINER-NAME> -v path/to/local/data:/data/data IMAGE-NAME
```
Flags:
- --rm: remove the container when execution ends. (optional)
- -i: set interactive mode. It is required to use standard input
- --name: a name for the container. (optional)
- v: volume binding. It maps a local directory to a directory inside the container so that local files can be accessed from it. The format is:
`/absolute/path/to/local/dir`:`/absolute/path/to/container/dir`


### Example
#### Execute container and access
Create a container and access the CLI, as specified in `Dockerfile`
```
docker run --rm -i --name container-name -v path/to/local/data/:/shared_data/ dom-class
```

#### Show help menu
```
docker run --rm -i --name container-name -v path/to/local/data/:/shared_data/ dom-class --help
```

### Execute task
```
docker run --rm -i --name container-name -v path/to/local/data/:/shared_data/ dom-class python main_dc_single_task.py --p /shared_data/projects/docker_proj --source /shared_data/datasets --task 
```
Alternatively, the command to execute the CLI can be executed first, and then the main command:
```
python main_dc_single_task.py --p /shared_data/projects/docker_proj --source /shared_data/datasets --task 
```
