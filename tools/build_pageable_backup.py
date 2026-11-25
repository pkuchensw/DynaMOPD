"""Build a job-local torch-memory-saver 0.0.9 with pageable weight backups."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

URL = ("https://files.pythonhosted.org/packages/28/6c/21dfda5d31afb71f52cedff52370acbb8290485b3f0fee6816a15a3d08f1/"
       "torch_memory_saver-0.0.9.tar.gz")
SHA256 = "3bbf76391fb16870b1b0df279fc281c8a05ef8f8809400b309b0a8240e8ee5ba"


def build(target, archive):
    data = archive.read_bytes() if archive else urlopen(URL,timeout=45).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:
        raise ValueError("source archive SHA256 mismatch")
    target = target.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
            bundle.extractall(stage,filter="data")
        root = stage/"torch_memory_saver-0.0.9"
        core = root/"csrc/core.cpp"
        source = core.read_text()
        replacements = {"CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));":
                            "metadata.cpu_backup = std::malloc(metadata.size);",
                        "CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));":"std::free(metadata.cpu_backup);"}
        for original,updated in replacements.items():
            if source.count(original)!=1:
                raise ValueError("unexpected memory saver source")
            source=source.replace(original,updated)
        core.write_text("#include <cstdlib>\n"+source)
        output = stage/"overlay"
        subprocess.run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output)], cwd=root, check=True)
        hook=next(output.glob("torch_memory_saver_hook_mode_preload.*.so"))
        if shutil.which("ninja"):
            subprocess.run(["ninja","--version"],env=dict(os.environ,LD_PRELOAD=str(hook)),check=True)
        shutil.copytree(root/"torch_memory_saver",output/"torch_memory_saver")
        shutil.copyfile(root/"LICENSE",output/"LICENSE")
        # The profile normally prepends a local torch/FA2 import cache. Retain it
        # while selecting only this memory saver package for driver and Ray workers.
        (output/"sitecustomize.py").write_text(
            "import os,runpy,sys\nfrom pathlib import Path\n"
            "profile=os.environ.get('DYNAMOPD_PROFILE_SITECUSTOMIZE')\n"
            "if profile:\n    runpy.run_path(profile)\n"
            "sys.path.insert(0,str(Path(__file__).resolve().parent))\n")
        (output/"SOURCE_SHA256").write_text(SHA256+"\n")
        output.rename(target)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--archive",type=Path)
    a=p.parse_args()
    build(a.out,a.archive)
