import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_type", type=str)
parser.add_argument("-c", "--config", type=str)
parser.add_argument("-s", "--stage", type=int)
args = parser.parse_args()

run_type = args.run_type
config:str = args.config
stage:int = args.stage

yaml_config = \
f'''
apiVersion: batch/v1
kind: Job
metadata:
  name: vthumuluri-job-{config.replace("_","-")}
  namespace: ai-md
  labels:
    user: vthumuluri
spec:
  ttlSecondsAfterFinished: 6000 # 100 minute to delete completed jobs
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-GeForce-RTX-2080-Ti
      containers:
      - name: gpu-container
        image: gitlab-registry.nrp-nautilus.io/vthumuluri/aimd
        command:
          - "sh"
          - "-c"
        args:
          - "cd /home/AIMD && conda init && . /opt/conda/etc/profile.d/conda.sh && conda activate pytorch && python limo.py --config {config} --start_stage {stage}"
        resources:
          requests:
            cpu: "12"
            memory: "8Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "12"
            memory: "8Gi"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm 
        - name: vthumuluri-slow-vol-aimd
          mountPath: /home
      volumes:
      - name: dshm 
        emptyDir:
          medium: Memory
      - name: vthumuluri-slow-vol-aimd
        persistentVolumeClaim:
          claimName: vthumuluri-slow-vol-aimd
      restartPolicy: Never
  backoffLimit: 0
'''

with open("temp_file.yaml", "w") as f:
	f.write(yaml_config)
os.system(f'kubectl {run_type} -f temp_file.yaml')
#os.system(f'rm temp_file.yaml')


