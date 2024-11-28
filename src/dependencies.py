#Importaciones para la vision computacional
import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt 
import time
import mediapipe as mp 
import sklearn


from draw_landmarks import draw_landmarks 
from mediapipe_dection import mediapipe_detection
from extract_keypoints import extract_keypoints


import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI 
from pydantic import BaseModel
import json
from typing import List

from dotenv import load_dotenv