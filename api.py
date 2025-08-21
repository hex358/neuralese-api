
import os
from dotenv import load_dotenv; load_dotenv()
print(os.getenv("ABCD"))

from sanic import Sanic
from sanic.request import Request
from sanic.response import text, html, json
from sanic.exceptions import NotFound, SanicException
import json as _json
import nns.graph_core as graph_core
import nns.model_core as nodes
import db.lset as lset

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

app = Sanic("neuralese_api")

res_dict: dict = {}

import gzip
@app.post("/run")
async def run_graph(request: Request):
    graph = _json.loads(gzip.decompress(request.body))
    nodes.load_model(graph_core.sh_context, "digit2")
    if graph["train"]:
        nodes.train(graph, graph_core.sh_context, 20, "datasets/mnist_noisy.ds", "digit2")
    else:
        # TODO: remove this nasty workaround
        inbox = graph_core.execute_graph(graph)
        nodes.load_model(graph_core.sh_context, "digit2")
        inbox = graph_core.execute_graph(graph)
        print(inbox.last_inbox)
    #inbox = graph_core.execute_graph(graph)
    #print(inbox.inbox_by_page)
    return json({})


if __name__ == "__main__":
    app.run(
        host="::",
        port=8100,
        debug=True,
    )
