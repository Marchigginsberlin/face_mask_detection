# Data analysis
- Document here the project: face_mask_detection
- Description: We created a video face mask detection for webcam video feeds.
- Data Source: We used various different datasets to provide our model with sufficient information.
      1) [Kaggle] (https://www.kaggle.com/vijaykumar1799/face-mask-detection) Dataset provided by vijay kumar
      2) [MaskedFace-Net] (https://github.com/cabani/MaskedFace-Net) a dataset of human faces with a correctly or incorrectly worn mask based on the dataset [Flickr-Faces-HQ] (https://github.com/NVlabs/ffhq-dataset)

- Challange: Creation of a fast and reliable deep learning classifier which is able to run in the browser.


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for face_mask_detection in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/face_mask_detection`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "face_mask_detection"
git remote add origin git@github.com:{group}/face_mask_detection.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
face_mask_detection-run
```

# Install

Go to `https://github.com/{group}/face_mask_detection` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/face_mask_detection.git
cd face_mask_detection
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
face_mask_detection-run
```
