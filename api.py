from sanic import Sanic
from sanic.request import Request
from sanic.response import text, html, json
from sanic.exceptions import NotFound, SanicException
import json as pyjson
import graph_core
import node_types


app = Sanic("neuralese_api")

res_dict: dict = {}

import gzip
@app.post("/run")
async def run_graph(request: Request):
    graph = pyjson.loads(gzip.decompress(request.body))
    print(graph_core.execute_graph(graph).last_inbox)
    return json({})


if __name__ == "__main__":
    app.run(
        host="::",
        port=8100,
        debug=True,
    )
