# Get all deployment names in the current namespace and restart each
for deployment in $(kubectl get deployments -o name -n monitoring)
do
  kubectl rollout restart $deployment
  echo "Restarted $deployment"
done

