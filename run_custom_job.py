import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run_type", type=str)
parser.add_argument("-c", "--config", type=str)
parser.add_argument("-m", "--model_type", type=str)
parser.add_argument("-s", "--start_stage", type=int)
parser.add_argument("-e", "--end_stage", type=int, default=10)
args = parser.parse_args()

run_type = args.run_type
config:str = args.config
start_stage:int = args.start_stage
end_stage:int = args.end_stage
model_type:str = args.model_type

yaml_config = \
f'''
apiVersion: batch/v1
kind: Job
metadata:
  name: vthumuluri-job-{config.replace("_","-")}-{model_type}
  namespace: ai-md
  labels:
    user: vthumuluri
spec:
  ttlSecondsAfterFinished: 36000 # 100 minute to delete completed jobs
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
                #- NVIDIA-A10
                - NVIDIA-GeForce-RTX-2080-Ti
      containers:
      - name: gpu-container
        image: gitlab-registry.nrp-nautilus.io/vthumuluri/aimd
        command:
          - "sh"
          - "-c"
        args:
          - "cd /home/AIMD && conda init && . /opt/conda/etc/profile.d/conda.sh && conda activate pytorch && python limo.py --config {config} --start_stage {start_stage} --end_stage {end_stage} --model_type {model_type}"
        resources:
          requests:
            cpu: "12"
            memory: "8Gi"
            nvidia.com/gpu: 1
            ephemeral-storage: 10Gi
          limits:
            cpu: "12"
            memory: "8Gi"
            nvidia.com/gpu: 1
            ephemeral-storage: 10Gi
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


