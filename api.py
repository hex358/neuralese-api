from sanic import Sanic
from sanic.request import Request
from sanic.response import text, html, json
from sanic.exceptions import NotFound, SanicException
import json as pyjson
import graph_core
import node_types

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

app = Sanic("neuralese_api")

res_dict: dict = {}

import gzip
@app.post("/run")
async def run_graph(request: Request):
    graph = pyjson.loads(gzip.decompress(request.body))
    node_types.train(graph, 100)
    #inbox = graph_core.execute_graph(graph)
    #print(inbox.inbox_by_page)
    return json({})


if __name__ == "__main__":
    app.run(
        host="::",
        port=8100,
        debug=True,
    )
