from clipper_admin import ClipperConnection, KubernetesContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model, create_endpoint
from clipper_admin.container_manager import CLIPPER_DOCKER_LABEL

import os
import argparse
import base64
import time
import logging
import docker
import requests
import json

logger = logging.getLogger(__name__)


input_tensor_name = 'input_example_tensor:0'
output_tensor_name = ['predictions/topk_y_pred:1', 
                      'predictions/topk_y_pred:0']

app_name = "pwc-adrank-app"
model_name = "pwc-adrank-model"


def _predict(sess, inp):
    inp = map(lambda e: base64.b64decode(e), inp)
    preds = sess.run(output_tensor_name,
                     feed_dict={input_tensor_name: inp})
    preds = zip(preds[0], preds[1])
    return preds


def deploy_and_test_model(clipper_conn, 
                          sess,
                          version,
                          input_type,
                          registry=None,
                          link_model=False,
                          only_test=False,
                          predict_fn=_predict):
   
    #old_version = clipper_conn.get_current_model_version(model_name)
    #if old_version != str(version):    
    #    print("{}:{} model deploying..".format(model_name, version)) 
    if not only_test :
        deploy_tensorflow_model(clipper_conn, model_name, 
                                version, input_type,
                                predict_fn, sess, 
                                registry=registry)
    if link_model:
        clipper_conn.link_model_to_app(app_name, model_name)
        time.sleep(5)

    if only_test:
        test_model(clipper_conn, app_name, version)


def test_model(clipper_conn, app_name, version):
    sparse_data = "CjMKGgoGdmFsdWVzEhASDgoMGHIxPxhyMT8YcjE/ChUKB2luZGljZXMSChoICgaNZPI2wWA="
    addr = clipper_conn.get_query_addr()
    url = "http://{}/{}/predict".format(addr, app_name)
    print(url)
    headers = {"Content-type": "application/json"}
    result = requests.post(
	"http://{}/{}/predict".format(addr, app_name), 
        headers=headers, 
        data=json.dumps({"input": sparse_data})).json()

    print(result)
    

def _get_lastest_model_version(export_dir):

    def is_integer(s):
        try:
            int(s)
            return True
        except:
            return False

    entries = [f for f in os.listdir(export_dir) if not os.path.isfile(f) and is_integer(f)]
    entries = map(lambda x: int(x), entries)
    entries.sort()

    return entries[-1]


def check_model(clipper_conn, version):
    model = clipper_conn.get_linked_models(app_name=app_name)
    print(model)
    num = clipper_conn.cm.get_num_replicas(name=model[0],
                                           version=version)
    if num == 0:
        print("model container crashed!")
    else:
        print("{} replicas  ok!".format(num))


def get_docker_client():
    if "DOCKER_API_VERSION" in os.environ:
        return docker.from_env(version=os.environ["DOCKER_API_VERSION"])
    else:
        return docker.from_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kube_ip", type=str, required=True) 
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--docker_registry", type=str, default="docker.io/yrbahn")
    parser.add_argument("--start_clipper", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--link_model", action="store_true")
    parser.add_argument("--only_test", action="store_true")

    #parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    cm = KubernetesContainerManager(args.kube_ip,
	useInternalIP=True)
    clipper_conn = ClipperConnection(cm)

    if args.cleanup:
        clipper_conn.stop_all()
        docker_client = get_docker_client()
        docker_client.containers.prune(filters={"label": CLIPPER_DOCKER_LABEL})
    
    if args.start_clipper:
        logger.info("Starting Clipper")
        clipper_conn.start_clipper()
        time.sleep(1)
    else:
        clipper_conn.connect()

    #register app
    if not args.only_test :
        clipper_conn.register_application(app_name,
                                          "string",
                                          "-1",
                                          1000000)
    #version
    version = _get_lastest_model_version(args.export_dir)
    print(version)

    #cd model_dir    
    os.chdir(args.export_dir)
    current_dir = os.getcwd()
    print(current_dir)
    
    sess = str(version) 

    deploy_and_test_model(clipper_conn,
                          sess,
                          version, 
                          "strings",
                          registry=args.docker_registry,
                          link_model=args.link_model,
                          only_test=args.only_test)
    
    #check_model(clipper_conn, version)
    
if __name__ == "__main__":
    main()

