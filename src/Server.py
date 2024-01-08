import flask
from flask import request

app = flask.Flask(__name__)

port_number = 8000


@app.route('/', methods=['GET', 'POST'])
def handle_resource():
    if request.method == 'POST':
        # Create a new resource
        data = request.get_json()
        print('Received POST request with data:', data)
        return 'Resource created successfully!', 201
    else:
        # Retrieve a list of resources
        return 'Retrieved a list of resources!', 200


if __name__ == '__main__':
    app.run(debug=True, port=port_number)
