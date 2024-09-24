set -x
./togglewasm true
sh restart-addtocart-svc.sh
python run_onlineboutique.py 50 jumping-2hr-varywest-92bg
./togglewasm false
sh restart-addtocart-svc.sh
python run_onlineboutique.py 50 nojumping-2hr-varywest-92bg
