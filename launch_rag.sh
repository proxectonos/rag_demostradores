#!/bin/bash

CONFIG=/home/compartido/pabloF/rag_demostradores/demostrador/backend/configs/general_config.json

source /home/compartido/pabloF/venvs/demo-rag/bin/activate

python3 -m demostrador.interfaz.rag_interface $CONFIG