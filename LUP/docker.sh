cd $(dirname $0)
pwd
#echo "pyenv" > .dockerignore
docker build -t mjw:lup-person-lup -f main.Dockerfile .
