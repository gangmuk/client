 cluster=$1
#method=$2

if [ -z "$cluster" ]; then
   echo "Usage: $0 <cluster>"
   echo "Example: ./curl.sh west"
   exit 1
fi

nodename=$(kubectl get nodes | grep "node7" | awk '{print $1}')
ingressgw_http2_nodeport=$(kubectl get svc istio-ingressgateway -n istio-system -o=json | jq '.spec.ports[] | select(.name=="http2") | .nodePort')
server_ip="http://${nodename}:${ingressgw_http2_nodeport}"
echo server_ip: $server_ip

#curl -v -XPOST -H "x-slate-destination: ${cluster}" "${server_ip}/cart/empty"

# curl -v -XPOST -H "x-slate-destination: ${cluster}" "${server_ip}/cart?product_id=OLJCESPC7Z&quantity=5"

# checkout cart
curl -v -XPOST -H "x-slate-destination: ${cluster}" "${server_ip}/cart/checkout?email=fo%40bar.com&street_address=405&zip_code=945&city=Fremont&state=CA&country=USA&credit_card_number=5555555555554444&credit_card_expiration_month=12&credit_card_expiration_year=2025&credit_card_cvv=222"

echo

exit
