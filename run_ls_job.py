import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str)
parser.add_argument("-m", "--model_type", type=str)
parser.add_argument("-n", "--exp_name", type=str, default="default")
args = parser.parse_args()

yaml_config = \
f'''
apiVersion: batch/v1
kind: Job
metadata:
  name: vthumuluri-job-ls-{args.config}-{args.model_type.replace("_","-")}-{args.exp_name}
  namespace: ai-md
  labels:
    user: vthumuluri
spec:
  ttlSecondsAfterFinished: 36000 # 100 minute to delete completed jobs
  template:
    spec:
      # affinity:
      #   nodeAffinity:
      #     requiredDuringSchedulingIgnoredDuringExecution:
      #       nodeSelectorTerms:
      #       - matchExpressions:
      #         - key: nvidia.com/gpu.product
      #           operator: In
      #           values:
      #           - NVIDIA-A10
                #- NVIDIA-GeForce-RTX-2080-Ti
      containers:
      - name: gpu-container
        image: gitlab-registry.nrp-nautilus.io/vthumuluri/aimd
        command:
          - "sh"
          - "-c"
        args:
          - "cd /home/AIMD && conda init && . /opt/conda/etc/profile.d/conda.sh && conda activate pytorch && pip install cmake cvxpy faiss-gpu && pip install loss-landscapes && python analyze_latent_space.py --config {args.config} --model_type {args.model_type} --exp_name {args.exp_name}"
        resources:
          requests:
            cpu: "12"
            memory: "16Gi"
            nvidia.com/gpu: 0
            ephemeral-storage: 10Gi
          limits:
            cpu: "12"
            memory: "16Gi"
            nvidia.com/gpu: 0
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
os.system(f'kubectl create -f temp_file.yaml')
#os.system(f'rm temp_file.yaml')


