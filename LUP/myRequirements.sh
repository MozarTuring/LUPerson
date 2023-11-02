cd $(dirname $0)
pwd
if [ ! -d "pyenv" ]; then
    /usr/local/python3.8.16/bin/python3 -m virtualenv pyenv
fi
source pyenv/bin/activate

#python -m pip install numpy
#python -m pip install tqdm
#curl -L https://yt-dl.org/downloads/latest/youtube-dl -o youtube-dl
#chmod a+rx youtube-dl
python -m pip install yt-dlp
