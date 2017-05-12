image_name = rfcn-docker
container_name = rfcn-docker

open_ports = 8000:8000

mounted_volumes = /home/ax:/home/user

publish = $(foreach port,$(open_ports),--publish $(port))
mount = $(foreach volume,$(mounted_volumes),-v $(volume))


.PHONY: build
build: Dockerfile
	docker build -t $(image_name) .


.PHONY: run
run: rm
	nvidia-docker run --interactive --detach=true $(publish) $(mount) --name $(container_name) $(image_name)


.PHONY: exec
exec:
	nvidia-docker exec -it $(container_name) bash


.PHONY: stop
stop:
	-docker stop $(container_name)


.PHONY: rm
rm: stop
	-docker rm $(container_name)
