import sys
import logging
from multiprocessing import cpu_count
import os
from dotenv import load_dotenv 
import numpy as np
import torch
from vc_infer_pipeline import VC
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from lib.audio import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile
import boto3

# Load environment variables
load_dotenv()

amazon_key = os.getenv("AMAZON_KEY")
amazon_secret = os.getenv("AMAZON_SECRET")

# Device setup
device = "cuda:0"
is_half = False

# AWS S3 setup
s3_client = boto3.client('s3', aws_access_key_id=amazon_key,
                        aws_secret_access_key=amazon_secret, region_name='us-east-2')
bucket_name = 'arn:aws:s3:us-east-2:015355617133:accesspoint/vclone-artifacts-endpoint'

# Download model from S3 to local
def download_model_from_s3(model_name):
    s3_key = f"models/{model_name}.pth"
    local_path = f"./artifacts/{model_name}.pth"
    s3_client.download_file(bucket_name, s3_key, local_path)
    return local_path


class Config:
    def __init__(self,device,is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

config=Config(device,is_half)
now_dir=os.getcwd()
sys.path.append(now_dir)

hubert_model=None
def load_hubert(path):
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([f"{path}"],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(
    sid=0,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="harvest",
    #file_index=None,  # .index files
    file_index2="",
    # file_big_npy,
    index_rate=1.0,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1.0,
    model_name="Joel",
    output_path="./audio_samples/test.wav",
    protect=0.33,
):
    model_path, index_path, hubert_path = get_model(model_name)

    file_index = index_path

    global tgt_sr, net_g, vc, hubert_model, version
    get_vc(model_path)
    if input_audio_path is None:
        return "You need to upload an audio file", None

    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio_path, 16000)
    audio_max = np.abs(audio).max() / 0.95

    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]

    if hubert_model == None:
        load_hubert(hubert_path)

    if_f0 = cpt.get("f0", 1)

    file_index = (
        (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if file_index != ""
        else file_index2
    )

    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        f0_file=f0_file,
        crepe_hop_length=160,
        protect=protect,
    )
    wavfile.write(output_path, tgt_sr, audio_opt)

    # Delete the temporary files
    os.remove(model_path)
    os.remove(index_path)
    os.remove(hubert_path)

    return output_path, audio_opt 


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    #elif version == "v2":
    #    if if_f0 == 1:
    #        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
    #    else:
    #        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}

def get_model(model_name:str="Joel"):
    try:
        # Decide the S3 path based on user-selected model name
        if model_name == "Joel":
            pth_path = "joel11320.pth"
            index_path = "s3://vclone-artifacts/trained_IVF479_Flat_nprobe_1.index"
            hubert_path = "hubert_base.pt"
        elif model_name == "Jenny":
            pth_path = "jenny3450.pth"
            index_path = "added_IVF533_Flat_nprobe_1.index"
            hubert_path = "hubert_base.pt"
        else:
            raise ValueError("Invalid model name")
    except Exception as e:
        print(e)
        raise e
    
    # Download from S3 to local (SageMaker instance)
    s3_client.download_file(bucket_name, pth_path, 'model.pth')
    s3_client.download_file(bucket_name, index_path, 'index.index')
    s3_client.download_file(bucket_name, hubert_path, 'hubert_base.pt')

    #Load the model and other files
    new_model_path = 'model.pth'
    new_index_path = 'index.index'
    new_hubert_path = 'hubert_base.pt'
    
    return new_model_path, new_index_path, new_hubert_path

    
        
input_audio_path = "./audio_samples/jenny_fcar.wav"
model_name = "Jenny"
output = vc_single(input_audio_path=input_audio_path, model_name=model_name)