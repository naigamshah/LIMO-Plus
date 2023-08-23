import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--type', type=str, default="surrogate")
parser.add_argument('--start_idx', type=int, default=-1)
args = parser.parse_args()

yaml_config = \
f'''
apiVersion: batch/v1
kind: Job
metadata:
  name: vthumuluri-job-losslandscape-{args.type}-{args.seed}-{args.start_idx}
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
                - NVIDIA-A10
                #- NVIDIA-GeForce-RTX-2080-Ti
      containers:
      - name: gpu-container
        image: gitlab-registry.nrp-nautilus.io/vthumuluri/aimd
        command:
          - "sh"
          - "-c"
        args:
          - "cd /home/AIMD && conda init && . /opt/conda/etc/profile.d/conda.sh && conda activate pytorch && pip install cmake cvxpy && pip install loss-landscapes && python plot_loss_landscapes.py --seed {args.seed} --type {args.type} --start_idx {args.start_idx}"
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
os.system(f'kubectl create -f temp_file.yaml')
#os.system(f'rm temp_file.yaml')


