#!/bin/bash
if [ "$1" = "gapfill" ] 
then
	gcloud container clusters get-credentials services-cluster -z asia-southeast1-b
	TAG="$(docker images -q gcr.io/questo2/gapfill)"
	docker tag ${TAG} gcr.io/questo2/gapfill:${TAG}
	docker tag ${TAG} gcr.io/questo2/gapfill:latest
	docker push gcr.io/questo2/gapfill:${TAG}
	docker push gcr.io/questo2/gapfill:latest
	kubectl set image deployment/gapfill-deployment gapfill=gcr.io/questo2/gapfill:${TAG}
elif [ "$1" = "template" ]
then
	gcloud container clusters get-credentials qg-cluster -z asia-southeast1-b
	TAG="$(docker images -q gcr.io/questo2/template)"
	docker tag ${TAG} gcr.io/questo2/template:${TAG}
	docker tag ${TAG} gcr.io/questo2/template:latest
	docker push gcr.io/questo2/template:${TAG}
	docker push gcr.io/questo2/template:latest
	kubectl set image deployment/template-deployment template=gcr.io/questo2/template:${TAG}
fi
