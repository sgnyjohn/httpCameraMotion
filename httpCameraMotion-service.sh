#!/bin/bash
# /lib/systemd/system/camera.service
# systemctl daemon-reload
# systemctl enable camera

lg=/var/log/camera-service.log
run=/var/run/camera-service.run
#########################################
log() {
	dt=$(date "+%Y-%m-%d %H:%M %a")
	echo "$dt $1" >>$lg
}

#log
log "inicio $1=$1 $2=$2"

monit() {
	local n=0
	local p
	while [ 1 -eq 1 ]; do 
		p=
		if [ $[n%9] -eq 0 ]; then
			p=$(ps aux|grep python|grep camera-srv|grep -v serv)
		fi
		log "rodando $n $p"
		sleep 20s
		let n=n+1
	done
}

if [ "$1" == "start" ]; then
	#python
	cd "$(dirname "$0")/www"
	su - alarm -c "cd /home/alarm/cam;python camera-srv.py >>log.txt" &
	echo "$!" >$run
	
	#monitor
	cd ..
	bash $0 monit &
	echo "$!" >>$run
	exit 0
fi

if [ "$1" == "monit" ]; then
	monit
fi

if [ "$1" == "stop" ]; then
	if test -e $run; then
		log "parando pids=($(cat $run))"
		for i in $(cat $run); do
			kill $i
		done
		rm $run
	fi
	exit 0
fi

