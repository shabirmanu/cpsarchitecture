from flask import Flask, render_template,  flash, redirect, url_for, session, request, make_response, jsonify
from flask_mysqldb import MySQL
from forms import *
import random,os, csv
from tasks import TaskType, TaskIns
from prime import *
import numpy as np
import json
import datetime
import urllib
import requests
from textblob import TextBlob, Word
from werkzeug.utils import  secure_filename
import pandas as pd
import xml.etree.ElementTree as ET
import html.parser
import sklearn

h = html.parser.HTMLParser

import argparse
import sys
import time
#import pallete


def connection():
    conn = MySQL.connect(host="localhost", user="root", passwd = "", db="mdevicemtask")
    c = conn.cursor()
    return c, conn


app = Flask(__name__)
app.debug = True

UPLOAD_FOLDER = 'static\\datasets\\'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xls', 'xlsx'}

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "mdmt"
app.config['MYSQL_CURSORCLASS'] = "DictCursor"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mysql = MySQL(app)

app.config.from_object('config')

data=[]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
class AddMicroService(Form):
    title = StringField('Name', [validators.required("Please enter title for the micro service.")])
    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name from problems")
        choices = cur.fetchall()

    choices_data = [(-1,'Select Problem')]
    for i in choices:
        choices_data += [(i['id'],i['name'])]
        print(i['id'],i['name'])

    service = SelectField('Select Service',choices=choices_data,id="select_service_id"
    )
    description = TextAreaField('Description', [validators.required("Please enter Description for the service.")])

class AddTask(Form):
    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name from problems")
        choices = cur.fetchall()

    choices_data = [(-1,'Select Service')]
    for i in choices:
        choices_data += [(i['id'],i['name'])]

    service = SelectField(
        'Select Service',
        choices=choices_data,
        id="select_service"
    )
    microservice = SelectField(
        'Select Micro Service',
        choices=[(-1,'Select Microservice')],
        id="select_mservice"
    )
    title = StringField('Title', [validators.required("Please enter title for the task.")])
    period = IntegerField('Period', [validators.required("Please enter period of the task.")])
    arrivalTime = IntegerField('Please enter Arrival Time Range', [validators.required("Please enter Arrival time End")])
    execution = IntegerField('Execution Time', [validators.required("Please enter Execution time")])
    deadline =      IntegerField('Deadline (To)', [validators.required("Please enter Deadline")])
    out_maxthreshold = IntegerField('Sensing Maximum Threshold',
                               [validators.required("Sensing Maximum Threshold")])
    out_minthreshold = IntegerField('Sensing Minimum Threshold', [validators.required("Please enter minimum Sensing threshold time")])
    period_maxthreshold = IntegerField('Period Maximum Threshold', [validators.required("Please enter maximum period threshold")])
    period_minthreshold = IntegerField('Period Minimum Threshold', [validators.required("Please enter minimum period threshold")])
    isEvent     = BooleanField( 'Event Driven Task')
    OperationMode     = BooleanField( 'Check for Observing Mode')
    #cur.close()


@app.route('/save-configuration')
def saveConf():
    return "saved"


@app.route('/gendataset', methods=['GET','POST'])
def genDataset():
    bme_url = "http://192.168.1.37/read-sensor?task=getTemperature"
    dust_url = "http://192.168.1.37/dust-sensor"
    co2_url = "http://192.168.1.80/getCo2"
    f = requests.get(bme_url)
    myfile = f.read()
    print(myfile)



@app.route('/deploy-tasks')
def deployTasks():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM mapped_task mt inner join virtualobjects as vo on vo.id = mt.vo_id inner join tasks t on t.id = mt.task_id  ")
    rows = cur.fetchall()
    return render_template("deploy_tasks.html", **locals())


@app.route('/deploy-solutions')
def deploySolutions():
    cur = mysql.connection.cursor()
    cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, p.name as prediction_name FROM `solution` ps INNER JOIN prediction p on ps.target1_id=p.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id where ps.solution_type=%s",('Prediction',))
    prows = cur.fetchall()

    cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, of.name as objective_name FROM `solution` ps INNER JOIN objective_function of on ps.target1_id=of.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id where ps.solution_type=%s",('Optimization',))
    orows = cur.fetchall()

    cur.execute(
        "SELECT *, mp.name as mp_name, st.name as solver_name FROM `solution` ps INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id where ps.solution_type=%s",('Scheduling',))
    srows = cur.fetchall()

    cur.execute(
        "SELECT *, mp.name as mp_name, st.name as solver_name FROM `solution` ps INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id where ps.solution_type=%s",
        ('Control',))
    crows = cur.fetchall()
    return render_template("deploy_solutions.html", **locals())

# Utility function to build constraints string from json
def _buildConstraintsString(constraints):

    dd = "\\" + constraints[0].get('op')

    _str = constraints[0].get("eq") + dd + constraints[0].get("val")
    return _str

@app.route('/view-solution', methods=["GET", "POST"])
def viewSolution():
    sid = request.args.get('solution_id', '01', type=str)
    stype = request.args.get('solution_type', '01', type=str)
    cur = mysql.connection.cursor()
    flag=True
    if(stype == 'Prediction'):
        cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, p.name as target_name FROM `solution` ps INNER JOIN prediction p on ps.target1_id=p.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id INNER JOIN datasets dt on p.dataset_id=dt.id where ps.solution_type=%s and ps.id=%s",('Prediction', int(sid),))
        rows = cur.fetchone()
        hyps = json.loads(rows["hyperparameters"])
        sep = ","
    else:
        cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, of.name as target_name FROM `solution` ps INNER JOIN objective_function of on ps.target1_id=of.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id INNER JOIN datasets dt on of.dataset_id=dt.id where ps.solution_type=%s and ps.id=%s",('Optimization',int(sid),))
        flag=False
        rows = cur.fetchone()
        # hyps = json.loads(rows["hyperparameters"])
        # dps = json.loads(rows["design_parameters"])
        # constraints = json.loads(rows["constraints"])
        # cons_str = _buildConstraintsString(constraints)
        sep ="\t"

    if(rows['file_name'] != ''):
        graphData = _parse_graphdata(rows['file_name'],sep)

    tmp = graphData

    return render_template("view_solution.html", **locals())


@app.route('/submit-solution', methods=["GET", "POST"])
def submitSolution():
    sid = request.args.get('id', '01', type=str)
    stype = request.args.get('stype', '01', type=str)
    cur = mysql.connection.cursor()
    flag=True
    if(stype == 'Prediction'):
        cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, p.name as target_name FROM `solution` ps INNER JOIN prediction p on ps.target1_id=p.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id INNER JOIN datasets dt on p.dataset_id=dt.id where ps.solution_type=%s and ps.id=%s",('Prediction', int(sid),))
        rows = cur.fetchone()
        res = runPredictionSolution(rows)
        return res
    else:
        cur.execute("SELECT *, mp.name as mp_name, st.name as solver_name, of.name as target_name FROM `solution` ps INNER JOIN objective_function of on ps.target1_id=of.id INNER JOIN solver_types st on ps.solver_id=st.id inner join microproblems mp on ps.mp_id = mp.id INNER JOIN datasets dt on of.dataset_id=dt.id where ps.solution_type=%s and ps.id=%s",('Optimization',int(sid),))
        flag=False
        rows = cur.fetchone()
        mod = _makeModFile(rows)
        data = _makeDatafile(rows)
        xmldata = _makeXML(mod,data)
        results = submitJobToNeos('shabirahmad87','571466@Neos',xmldata)
        print(results)
        return jsonify(results)


def runPredictionSolution(row):

    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as met
    from math import sqrt
    from sklearn.metrics import r2_score, mean_squared_error

    if(row['name'] == 'ANN'):
        name="neural_network"
    hyp = json.loads(row['hyperparameters'])

    hL = next((item for item in hyp if item["name"] == "hidden_layers"), None)
    type = next((item for item in hyp if item["name"]=="type"),None)

    data = getFrameFromFile(row['file_name'])

    out_col = data.iloc[:, -1]
    out_col = 5*out_col
    in_col_length = len(data.columns) - 2
    in_col = data.iloc[:,0:in_col_length]

    X = in_col.values
    y = out_col.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    print(X_train.shape)
    print(X_test.shape)

    mlp = MLPRegressor(activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train, y_train)


    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(predict_train,predict_test)
    train_score = met.r2_score(y_train,predict_train)
    test_score = met.r2_score(y_test,predict_test)
    rmse_score = sqrt(mean_squared_error(y_test,predict_test))

    # in_col = []
    # for i in range(0, in_col_length):
    #     in_col.append(data.iloc[:, i].values.tolist())

    if(rmse_score <0.5):
        suggestion = "Accuracy is low: Use a different solver"
    elif(rmse_score >0.5 and rmse_score<0.7):
        suggestion = "Accuracy is Medium: Trying changing hyper-parameter"
    else:
        suggestion = "Accuracy is High: Solver is good to use"

    return jsonify({"train_r2_score":train_score, "test_r2_score":test_score, "RMSE":rmse_score, "suggestions":suggestion, "predicted":predict_test.tolist(), "actual":y_test.tolist()})





def _makeXML(mod, d='', cmd=''):
    xmldata = ET.Element('MyProblem')
    category = ET.SubElement(xmldata, 'category')
    category.text = 'nco'

    inputType = ET.SubElement(xmldata, 'inputType')
    inputType.text = 'AMPL'

    solver = ET.SubElement(xmldata, 'solver')
    solver.text = 'minos'



    priority = ET.SubElement(xmldata, 'priority')
    priority.text = 'long'

    email = ET.SubElement(xmldata, 'email')
    email.text = 'shabir@jejunu.ac.kr'

    model = ET.SubElement(xmldata, 'model')
    model.text = '<![CDATA['+mod+']]>'

    if(d != ''):
        data = ET.SubElement(xmldata, 'data')
        data.text = '<![CDATA[' + d + ';]]>'

    commands = ET.SubElement(xmldata, 'commands')
    commands.text = '<![CDATA[]]>'

    xml_file =  ET.tostring(xmldata).decode("utf-8")
    myfile = open("problem.xml", "w")
    myfile.write(html.unescape(xml_file))

    return xml_file



def submitJobToNeos(username, password,xml,action=""):
    try:
        import xmlrpc.client as xmlrpclib
    except ImportError:
        import xmlrpclib

    # parser = argparse.ArgumentParser()
    # parser.add_argument("action", help="specify XML file name or queue for action")
    # parser.add_argument("--server", default="https://neos-server.org:3333", help="URL to NEOS Server XML-RPC interface")
    # parser.add_argument("--username", default=os.environ.get("NEOS_USERNAME", None),
    #                     help="username for NEOS Server user account")
    # parser.add_argument("--password", default=os.environ.get("NEOS_PASSWORD", None),
    #                     help="password for NEOS Server user account")
    # args = parser.parse_args()

    server = "https://neos-server.org:3333"

    neos = xmlrpclib.ServerProxy(server)

    alive = neos.ping()
    if alive != "NeosServer is alive\n":
        sys.stderr.write("Could not make connection to NEOS Server\n")
        sys.exit(1)
    action=""
    if action == "queue":
        msg = neos.printQueue()
        sys.stdout.write(msg)
    else:
        # try:
        #     xmlfile = open(args.action, "r")
        #     buffer = 1
        #     while buffer:
        #         buffer = xmlfile.read()
        #         xml += buffer
        #     xmlfile.close()
        # except IOError as e:
        #     sys.stderr.write("I/O error(%d): %s\n" % (e.errno, e.strerror))
        #     sys.exit(1)
        xml = html.unescape(xml)
        if username and password:
            (jobNumber, password) = neos.authenticatedSubmitJob(xml, username, password)
        else:
            (jobNumber, password) = neos.submitJob(xml)
        sys.stdout.write("Job number = %d\nJob password = %s\n" % (jobNumber, password))

        if jobNumber == 0:
            sys.stderr.write("NEOS Server error: %s\n" % password)
            sys.exit(1)
        else:
            offset = 0
            status = ""
            while status != "Done":
                time.sleep(1)
                (msg, offset) = neos.getIntermediateResults(jobNumber, password, offset)
                sys.stdout.write(msg.data.decode())
                status = neos.getJobStatus(jobNumber, password)
            msg = neos.getFinalResults(jobNumber, password)
            return msg.data.decode()
            #sys.stdout.write(msg.data.decode())


def _makeModFile(row):
    #hyps = json.loads(row["hyperparameters"])
    #dps = json.loads(row["design_parameters"])
    #constraints = json.loads(row["constraints"])
    #dps_assign = json.loads(row["design_assignment"])

    modStr = "# Design Variables \n"+row["design_parameters"]
    # for dv in dps:
    #     modStr += dv['type']+" "+dv['name']+";\n"

    # if not dps_assign:
    #     dps_assign=""
    # else:
    #     for dv in dps_assign:
    #         modStr += dv['var']+" "+ dv['op']+" "+dv['val']+";\n"

    modStr+="#Part 2: Objective Function \n"
    if(row['goal'] == 0):
        modStr+="minimize "+row['cost']+";\n"
    if (row['goal'] == 1):
        modStr += "maximize " + row['cost'] + ";\n"

    modStr += "#Part 3: CONSTRAINTS \n"+row['constraints']






    return modStr

# def _makeCommandFile(row):
#     cmdStr = "reset; \n"
#     cmdStr += "solve; \n"
#     dps = json.loads(row["design_parameters"])
#     cmdStr+="display "
#     for dp in dps:
#         if (dp['type'] == "var"):
#             cmdStr += dp['name']+","
#     return cmdStr

def _makeDatafile(rows):
    dataStr = rows['design_assignment']
    lines = dataStr.split(";")
    #assignments = dataStr.rsplit(":=")
    hL = next((item for item in lines if item.rsplit(":=")[1] == "!dataset"),None)
    if (hL is not None):
        datasetStr = _getFileContent(rows['file_name'])

    hLN = next((i for i,item in enumerate(lines) if item.rsplit(":=")[1] == "!dataset"), None)
    _tmp = lines[hLN].rsplit(":=")[1]
    lines[hLN].replace(_tmp, datasetStr)
    newlines = []
    i=0
    for line in lines:
        if(i==hLN):
            rhs=lines[hLN].rsplit(":=")[0]
            newlines.append(rhs+":="+datasetStr)
        else:
            newlines.append(line)
        i=i+1
    return ";".join(newlines)

def _getFileContent(file_name):
    file_location = UPLOAD_FOLDER + "\\" + file_name
    with open(file_location, 'r') as file:
        data = file.read()
    return data

def getFrameFromFile(file_name,sep=","):
    file_location = UPLOAD_FOLDER + "\\" + file_name
    data = pd.read_csv(file_location, sep=sep, skiprows=1, header=None)
    return data

def _parse_graphdata(file_name,sep=","):
    data = getFrameFromFile(file_name,sep)
    x_col = list(range(1,100))
    out_col = data.iloc[0:100,-1:].values.tolist()
    in_col_length = len(data.columns)-1
    in_col = []
    for i in range(0,in_col_length):
        in_col.append(data.iloc[0:100,i].values.tolist())


    return [x_col, in_col, out_col]

@app.route('/virtual-objects')
def virtualObjects():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM virtualobjects")
    rows = cur.fetchall()



    return render_template("virtual-objects.html", **locals())

@app.route('/map-tasks')
def mapTasks():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM virtualobjects")
    vos = cur.fetchall()

    cur.execute("SELECT * FROM tasks")
    tasks = cur.fetchall()

    return render_template("map_tasks.html", **locals())


@app.route('/cps-design')
def cpsDesign():
    proj_id = 1
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM problems where project_id=%s",(int(proj_id),))
    problems = cur.fetchall()

    mp_data = []
    for problem in problems:
        id = problem["id"];
        name = problem["name"];

        cur.execute("SELECT * FROM microproblems where service_id=%s", (int(id),))
        microproblems = cur.fetchall()
        mps = []
        mpids = []
        for microproblem in microproblems:
            cur.execute(
                "SELECT * FROM mapped_task mt inner join virtualobjects as vo on vo.id = mt.vo_id inner join tasks t on t.id = mt.task_id where t.ms_id = %s",
                (int(microproblem["id"]),))
            mappedtasks = cur.fetchall()
            mpids.append(microproblem["id"])
            mps.append({'mpid': microproblem["id"], 'mp_name':microproblem["name"], 'mptasks':mappedtasks})

        mp_data.append({'pid':id, 'title':name, 'microproblems':mps})



    cur.execute("SELECT * FROM objective_function")
    obfs = cur.fetchall()

    cur.execute("SELECT * FROM solver_types where algorithm=%s", ('Prediction',))
    pred_solvers = cur.fetchall()

    cur.execute("SELECT * FROM solver_types where algorithm=%s", ('Optimization',))
    opt_solvers = cur.fetchall()

    cur.execute("SELECT * FROM solver_types where algorithm=%s", ('Scheduling',))
    sch_solvers = cur.fetchall()

    cur.execute("SELECT * FROM solver_types where algorithm=%s", ('Control',))
    con_solvers = cur.fetchall()

    cur.execute("SELECT * FROM prediction")
    preds = cur.fetchall()

    cur.execute("SELECT * FROM tasks where ms_id = %s",(int(microproblem["id"]),))
    tasks = cur.fetchall()


    cur.execute("SELECT * FROM virtualobjects")
    vos = cur.fetchall()







    return render_template("cps-design.html", **locals())

@app.route('/')
def pindex():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM virtualobjects")
    vos = cur.fetchall()


    cur.execute("SELECT * FROM devices")
    devices = cur.fetchall()

    cur.execute("SELECT * FROM tasks")
    tasks = cur.fetchall()




    cur.execute("SELECT name,cost,goal FROM objective_function")
    ofs = cur.fetchall()


    of_data = []
    # for of in ofs:
    #     # cost = of["cost"]
    #     ddps = json.loads(of["design_parameters"])
    #
    #     constraints = json.loads(of["constraints"])
    #
    #     dd = "\\" + constraints[0].get('op')
    #
    #     _str = constraints[0].get("eq") + dd + constraints[0].get("val")
    #     if (of["goal"] == 0):
    #         _goal = "Minimize"
    #     _goal = "Maximize"
    #     of_data.append({"name": of["name"], "dp": ddps, "cost": of["cost"], "constraints": _str, "goal": _goal})

    cur.execute("SELECT * FROM solver_types")
    solvers = cur.fetchall()
    cur.close()

    # slv_data = []
    # for of in ofs:
    #     # cost = of["cost"]
    #     ddps = json.loads(of["hyperparameters"])
    #
    #     slv_data.append({"name": of["name"], "dp": ddps, "type": of["algorithm"]})

    return render_template("index.html", **locals())

@app.route('/addvirtualobj',methods=['GET','POST'])
def addVirtualObject():
    form = AddVirtualObject(request.form)
    if (request.method == 'POST' and form.validate()):
        name = form.name.data
        taskTags = json.dumps(form.taskTags.data)
        url = form.url.data
        methods = json.dumps(form.methods.data)
        attributes = json.dumps(form.attributes.data)



        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO virtualobjects(name, tasktags, url, methods, attributes) VALUES (%s, %s, %s, %s, %s)", (name, taskTags, url, methods, attributes))
        mysql.connection.commit()
        cur.close()

        flash('Task created successfully', 'success')


    return render_template("addvirtualobj.html",form=form)

@app.route('/addservice',methods=['GET','POST'])
def addService():
    form = AddService( request.form )
    if (request.method == 'POST' and form.validate()):
        title = form.title.data
        description = form.description.data

        cur = mysql.connection.cursor()

        now = datetime.datetime.now()

        cur.execute("INSERT INTO problems(name, description, created_at, updated_at) VALUES (%s, %s, %s, %s)", (title, description, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        inserted_id = cur.lastrowid
        cur.close()

        flash('Service created successfully', 'success')
        ms, tasks = _analyzeService(description)
        return render_template("addservice.html",service_id=inserted_id, ms=ms, tasks=tasks, suggestion = 1, form=form)

    return render_template("addservice.html",suggestion =0, form=form)



@app.route('/addproject',methods=['GET','POST'])
def addProject():
    form = AddProject( request.form )
    if (request.method == 'POST' and form.validate()):
        name = form.name.data
        description = form.description.data

        cur = mysql.connection.cursor()

        now = datetime.datetime.now()

        cur.execute("INSERT INTO project(name, description, created_at, updated_at) VALUES (%s, %s, %s, %s)", (name, description, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        inserted_id = cur.lastrowid
        cur.close()

        flash('Project created successfully', 'success')


    return render_template("addproject.html", form=form)


@app.route('/addsolver-type',methods=['GET','POST'])
def addSolverType():
    cur = mysql.connection.cursor()
    form = AddSolverType( request.form )
    if (request.method == 'POST' and form.validate()):
        name = form.name.data
        hyps = form.hyperparameters.data
        algorithms = form.algorithm.data

        lines = hyps.splitlines()

        hyperparameters = []
        for line in lines:
            hype_name = line.rsplit('|')[0]
            hype_value = line.rsplit('|')[1]
            hyperparameters.append({'name':hype_name, 'value':hype_value})

        hyperparameters = json.dumps(hyperparameters, indent=4)

        now = datetime.datetime.now()

        cur.execute("INSERT INTO solver_types(name, hyperparameters, algorithm, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)", (name, hyperparameters, algorithms, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        inserted_id = cur.lastrowid

        flash('Solver type created successfully', 'success')

    cur.execute("SELECT * FROM solver_types")
    ofs = cur.fetchall()
    cur.close()

    slv_data = []
    for of in ofs:
        # cost = of["cost"]
        ddps = json.loads(of["hyperparameters"])

        slv_data.append({"name": of["name"], "dp": ddps, "type": of["algorithm"]})

    return render_template("addsolver-type.html", **locals())



@app.route('/addof',methods=['GET','POST'])
def addObjectiveFunction():
    cur = mysql.connection.cursor()
    now = datetime.datetime.now()
    form = AddObjectiveFunction( request.form )
    requestsss = request
    if (request.method == 'POST' and form.validate()):
        name = form.name.data
        dsp = form.des_param.data
        dsp_assign = form.data_assign.data
        obf = form.obf.data
        constraints = form.constraints.data
        goal = form.goal.data

        if 'dataset' not in request.files:
            flash('No file part')
            return render_template("addobjective-function.html", **locals())

        file = request.files['dataset']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template("addobjective-function.html", **locals())
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

            cur.execute(
                "INSERT INTO datasets (file_name, created_at, updated_at) VALUES (%s, %s, %s)",
                (file.filename, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
            mysql.connection.commit()
            dataset_id = cur.lastrowid


        # dsp_dec_json = _parseDPJSON(dsp)
        # dsp_assign_json = _parseDPJSONAssign(dsp_assign)
        #
        #
        # clines = constraints.splitlines()
        # consvas = []
        # for line in clines:
        #     cst_eq = line.rsplit('|')[0]
        #     cst_op = line.rsplit('|')[1]
        #     cst_val = line.rsplit('|')[2]
        #
        #     consvas.append({'eq': cst_eq, 'op': cst_op, 'val':cst_val})
        #
        # consvas = json.dumps(consvas, indent=4)

        cur = mysql.connection.cursor()



        cur.execute("INSERT INTO objective_function(name, design_parameters, design_assignment, cost, constraints, goal, dataset_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (name, dsp, dsp_assign, obf, constraints, goal,dataset_id, now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        inserted_id = cur.lastrowid


        flash('Objective function created successfully', 'success')

    cur.execute("SELECT name, cost, goal FROM objective_function")
    ofs = cur.fetchall()
    cur.close()

    # of_data = []
    # for of in ofs:
    #     # cost = of["cost"]
    #     ddps = _isEmpty(of["design_parameters"])
    #
    #     ddps_assign = _isEmpty(of["design_assignment"])
    #
    #
    #
    #     constraints = json.loads(of["constraints"])
    #
    #
    #
    #     if(of["goal"] == 0):
    #         _goal = "Minimize"
    #     _goal = "Maximize"
    #     of_data.append({"name":of["name"], "dp":ddps , "dp_assign":ddps_assign, "cost":of["cost"], "constraints":constraints, "goal":_goal})


    return render_template("addobjective-function.html", **locals())



def _isEmpty(field):
    if not field:
        field = ""
        return field
    return json.loads(field)
# Utility function to parse design variable to json
def _parseDPJSON(field_val):
    lines = field_val.splitlines()
    designvas = []
    for line in lines:
        dsp_type = line.rsplit('|')[0]
        dsp_var = line.rsplit('|')[1]
        designvas.append({'type': dsp_type, 'name': dsp_var})

    designvas = json.dumps(designvas, indent=4)
    return designvas

# Utility function to parse design variable to json
def _parseDPJSONAssign(field_val):
    lines = field_val.splitlines()
    designvas = []
    for line in lines:
        dsp_type = line.rsplit('|')[0]
        dsp_var = line.rsplit('|')[1]
        dsp_op = line.rsplit('|')[2]
        dsp_val = line.rsplit('|')[3]
        designvas.append({'type':dsp_type,'var': dsp_var, 'op': dsp_op, 'val': dsp_val})

    designvas = json.dumps(designvas, indent=4)
    return designvas

@app.route('/addpred',methods=['GET','POST'])
def addPredictionObjective():
    cur = mysql.connection.cursor()
    now = datetime.datetime.now()
    form = AddPredictionObjectiveForm( request.form )
    requestsss = request
    if (request.method == 'POST' and form.validate()):
        name = form.name.data
        inf = form.in_features.data
        onf = form.op_features.data
        hyp = form.hyperparameter.data

        if 'dataset' not in request.files:
            flash('No file part')
            return render_template("addprediction-objective.html", **locals())

        file = request.files['dataset']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template("addprediction-objective.html", **locals())
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            cur.execute(
                "INSERT INTO datasets (file_name, created_at, updated_at) VALUES (%s, %s, %s)",
                (file.filename, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
            mysql.connection.commit()
            dataset_id = cur.lastrowid

        lines = inf.splitlines()
        infvas = []
        for line in lines:
            if_var = line.rsplit('|')[0]
            in_pos = line.rsplit('|')[1]
            infvas.append({'name': if_var, 'position': in_pos})

        infvas = json.dumps(infvas, indent=4)

        lines = onf.splitlines()
        onfvas = []
        for line in lines:
            of_var = line.rsplit('|')[0]
            on_pos = line.rsplit('|')[1]
            onfvas.append({'name': of_var, 'position': on_pos})

        onfvas = json.dumps(onfvas, indent=4)


        cur = mysql.connection.cursor()



        cur.execute("INSERT INTO prediction(name, in_features, op_feature, hyperparameter, dataset_id, created_at) VALUES (%s, %s, %s, %s, %s, %s)", (name, infvas, onfvas, hyp, dataset_id, now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        inserted_id = cur.lastrowid


        flash('Prediction Objective created successfully', 'success')

    cur.execute("SELECT * FROM prediction inner join datasets on prediction.dataset_id = datasets.id")
    ofs = cur.fetchall()
    cur.close()

    of_data = []
    for of in ofs:
        # cost = of["cost"]
        inf = json.loads(of["in_features"])

        opf = json.loads(of["op_feature"])


        of_data.append({"name":of["name"], "inf":inf ,"opf":opf, "hyper":of['hyperparameter'], "dataset":of['file_name']})


    return render_template("addprediction-objective.html", **locals())


@app.route('/addmicroproblem',methods=['GET','POST'])
def addMicroService():
    form = AddMicroService( request.form )

    # if request.method == 'GET':
    #     return render_template('addmicroservice.html', form=form)
    if (request.method == 'POST' ):
        service_id = form.service.data
        title = form.title.data
        description = form.description.data

        cur = mysql.connection.cursor()

        now = datetime.datetime.now()

        cur.execute("INSERT INTO microproblems(name, description, service_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)", (title, description, service_id, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
        cur.close()

        flash('MicroService created successfully', 'success')
        #_analyzeService(description)

    return render_template("addmicroservice.html",**locals())

# Some NLP for analyzing Service Requirement
def _analyzeService(text):
    goal_synonym = set(['goal', 'motivation', 'purpose', 'task', 'responsible for'])
    sensor_synonym = set(['sensor', 'sense', 'read'])
    periodic_synonym = set(['period', 'periodically', 'repeat', 'repeatedly'])
    blob = TextBlob(text)
    sentences = blob.sentences
    matches = [s for s in blob.sentences if goal_synonym & set(s.words)]
    if (len(matches) > 0):
        ms_sentence = matches[0]

    ms_index = sentences.index(ms_sentence)
    ms_sentences = sentences.pop(ms_index)

    if (len(matches) > 0):
        ms_sentence = matches[0]

    # Get Microservice Suggestions
    micro_services = []
    for ms in ms_sentences.noun_phrases:
        ms_arr = ms.split(" ")
        ms_arr_set = set(ms_arr)

        if goal_synonym & set(ms_arr_set):
            g_matches = goal_synonym & set(ms_arr_set)
            print(ms)
            print(g_matches)
            continue
        micro_services.append(ms)

    tasks = []
    for sentence in sentences:
        for np in sentence.noun_phrases:
            np_array = np.split(" ")
            # print(np_array[0],np_array[1])
            if (np_array[1] == "sensor"):
                verb = "get"
                task_name = verb + np_array[0]
            else:
                verb = Word(np_array[0]).lemmatize()
                task_name = np_array[0] + np_array[1]
            tasks.append(task_name)

    return micro_services, tasks



@app.route('/discard',methods=['POST','GET'])
def discardPost():
    tasks = request.args.get('tasks')
    ms = request.args.get('ms')
    index = request.args.get('tableIndex')
    flag = request.args.get('flag')

    if(flag):
        tasks.pop(index)
        msg = "Task"
    else:
        msg = "Microservice"
        ms.pop(index)

    return jsonify(tasks), jsonify(ms)



@app.route('/addtask',methods=['GET','POST'])
def addTask():
    form = AddTask( request.form )
    services = form.service.data
    microservices = form.microservice.data
    if request.method == 'GET':
        task_def_val = request.args.get('tname')
        ser_def_val = request.args.get('sid')
        form.title.data = task_def_val
        form.service.default = ser_def_val
        #form.service = str(ser_def_val)
        return render_template('addtask.html',form=form)
    if (request.method == 'POST'):
        title = form.title.data
        period = int (form.period.data)
        arrivalTime = int (form.arrivalTime.data)
        execution = int (form.execution.data)
        deadline = int (form.deadline.data)
        op_max = int (form.out_maxthreshold.data)
        op_min = int (form.out_minthreshold.data)
        period_max = int (form.period_maxthreshold.data)
        period_min = int (form.period_minthreshold.data)
        is_observing = bool (form.OperationMode.data)
        isEvent = form.isEvent.data

        if(isEvent == True):
            urgency = urgency = random.randint( int( 0 ), int( 1 ) )
        else:
            urgency = 'NA'

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO tasks(title, period, deadline, arrival, execution,ms_id, op_max, op_min, period_max, period_min, is_observing) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (title, period, deadline, arrivalTime, execution, microservices, op_max, op_min, period_max, period_min, is_observing))
        mysql.connection.commit()
        cur.close()

        flash('Task created successfully', 'success')


    return render_template("addtask.html",form=form)


# @app.route('/edittask/<int:task_id>', methods=['GET','POST'])
# def edit_task(task_id):
#     return render_template( "addtask.html", form=form )

@app.route('/_save_states/', methods=['GET','POST'])
def _save_states():
    tasks = request.get_json()
    #to_id = request.args.get('to', '01', type=str)


    cur = mysql.connection.cursor()
    now = datetime.datetime.now()

    for mline in tasks:
        cur.execute(
            "INSERT INTO mapped_task(task_id, vo_id, mapped_time) "
            "VALUES (%s, %s, %s)", (
            mline['from'], mline['to'], now.strftime("%Y-%m-%d %H:%M:%S")))
        mysql.connection.commit()
    cur.close()
    return jsonify(1)


@app.route('/_save_sch_solution/', methods=['GET','POST'])
def _save_sch_solution():
    from_id = request.args.get('from', '01', type=str)
    to1_id = request.args.get('to', '01', type=str)
    type = request.args.get('type', '01', type=str)

    cur = mysql.connection.cursor()
    now = datetime.datetime.now()

    # Optimization Solver
    cur.execute(
        "INSERT INTO solution(mp_id, target1_id, solver_id, solution_type, created_at) "
        "VALUES (%s, %s, %s, %s, %s)", (
        from_id, 0, to1_id, type, now.strftime("%Y-%m-%d %H:%M:%S")))
    mysql.connection.commit()
    cur.close()


    return jsonify(1)


@app.route('/_save_solution/', methods=['GET','POST'])
def _save_solution():
    from_id = request.args.get('from', '01', type=str)
    to1_id = request.args.get('opto1', '01', type=str)
    to2_id = request.args.get('opto2', '01', type=str)
    predto1_id = request.args.get('predto1', '01', type=str)
    predto2_id = request.args.get('predto2', '01', type=str)
    cur = mysql.connection.cursor()
    now = datetime.datetime.now()

    # Optimization Solver
    cur.execute(
        "INSERT INTO solution(mp_id, target1_id, solver_id, solution_type, created_at) "
        "VALUES (%s, %s, %s, %s, %s)", (
        from_id, to1_id, to2_id, 'Optimization', now.strftime("%Y-%m-%d %H:%M:%S")))
    mysql.connection.commit()
    cur.close()

    # Prediction Solver
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO solution(mp_id, target1_id, solver_id,solution_type, created_at) "
        "VALUES (%s, %s, %s, %s, %s)", (
            from_id, predto1_id, predto2_id, 'Prediction', now.strftime("%Y-%m-%d %H:%M:%S")))
    mysql.connection.commit()
    cur.close()

    return jsonify(1)


@app.route('/_get_microservices/')
def _get_microservices():
    service_id = request.args.get('service', '01', type=str)
    with app.app_context():
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, name from microproblems where service_id=%s",service_id)
        choices = cur.fetchall()

    choices_data = [(-1,'Select Microservice')]
    for i in choices:
        choices_data += [(i['id'],i['name'])]
        print(i['id'],i['name'])
    return jsonify(choices_data)

@app.route('/gentasks', methods=['GET','POST'])
def genTasks():
    form = GenerateTasks(request.form)
    if(request.method == 'POST' and form.validate()):
        noTasks = form.noOfTasks.data
        periodRangeFrom = form.periodRangeFrom.data
        periodRangeTo = form.periodRangeTo.data
        timeRangeFrom = form.timeRangeFrom.data
        timeRangeTo = form.timeRangeTo.data
        execRangeTo = form.execRangeFrom.data
        execRangeFrom = form.execRangeTo.data


        global data
        file_headers = ['Tasks No', 'Period', 'Execution Time', 'Start Time', 'Deadline', 'Urgency']
        data.append(file_headers)



        file = open( 'tasks.csv', 'w' , newline='')
        writer = csv.writer( file, delimiter=',', quotechar='"' )



        overhead = 0
        while(True):
            overhead += 1
            hyp = []
            data = []
            data.append( file_headers )
            for i in range( 0, int( noTasks ) ):
                period = random.randint( int( periodRangeFrom ), int( periodRangeTo ) )
                if(int(execRangeFrom) < period):
                    executiontime = random.randint( int( execRangeTo ), int(execRangeFrom) )
                else:
                    executiontime = random.randint( int( execRangeTo ), int( period ) )
                    executiontime = executiontime - 1

                starttime = 0
                deadline = period
                hyp.append(period)

                urgency = random.randint( int( 0 ), int( 1 ) )
                data.append([str(i+1), str(period), str(executiontime), str(starttime), str(deadline), str(urgency)])

            hyperperiod = lcm(hyp)
            print (hyperperiod)

            if(hyperperiod < 500):
                break


        print (overhead);
        print ('o====h')
        print (hyperperiod)


        writer.writerows(data)
        data = []
        file.close()

        flash("Tasks Generated Successfully!!",'success')
        redirect(url_for('index'))
    return render_template("gentasks.html", form=form)




@app.route('/edittask')
def editTask():
    hyperperiod, task_types, total_tasks = _tasksReaderDB( 'tasks.csv' )
    return render_template("edittask.html", **locals())

def getKey(obj):
    return obj.Tno

@app.route('/ratemonotonic')
def rateMono():

    page_title = "Arrival Based Scheduling"

    hyperperiod,task_types, total_tasks = _tasksReaderDB()

    html_color = {'Core1': 'red', 'Core2': 'blue', 'Core3': 'green', 'Core4': 'aqua', 'Empty': 'grey',
                  'Finish': 'black'}

    tasks = []
    line = []
    total = 0
    # Create task instances
    task_types = sorted(task_types, key=getKey)
    for i in np.arange( 0, int(hyperperiod) ):
        for task_type in task_types:

            if(task_type.period != 0 ):
                _cond = (int(i) - int(task_type.release)) % int(task_type.period)

                if (_cond == 0 and int(i) >= int(task_type.release)):
                    start = i
                    end = start + task_type.execution
                    priority = task_type.urgency
                    deadline = start + task_type.deadline
                    Task_no = task_type.Tno
                    vo_name = task_type.vo
                    # We make some attributes 0 because these are not required in this protocol
                    tasks.append( TaskIns( start=start, end=end, priority=priority, name=task_type.name, deadline=deadline,
                                           Tno=Task_no,pD=0, pP=0, pS=0, pE=0, urgency=0, vo=vo_name, pB=0) )
                    total = total + task_type.execution



    core = round( total / float( hyperperiod ) )


    print ("Least No of Cores Needed: " + str( core ))


    # Check if the tasks are schedulable

    process_utilization = total_tasks*(2**(1/total_tasks)-1)


    cpu_utilization = 0
    for task_type in task_types:
        if(task_type.period != 0):
            cpu_utilization += float( task_type.execution ) / float( task_type.period )

    if(cpu_utilization < process_utilization):
        print ("all tasks are schedulable")

    if cpu_utilization > 1:
        a = 1
        error = True
        error_msg = "utilization greater than 1, so some tasks will not be able to reach deadline"
       # return render_template( "ratemono.html", **locals() )

    hyperperiod = int (hyperperiod)

    clock_step = 1
    html = ''
    task_timeline = []

    cpu_tasks = []
    for i in np.arange( 0, hyperperiod, clock_step ):
        # Fetch possible tasks that can use cpu and sort by priority
        possible = []
        for t in tasks:
            if t.start <= i:
                possible.append( t )
        possible = sorted( possible, key=cmp_to_key(priority_cmp) )

        # Select task with highest priority

        if len( possible ) > 0:
            on_cpu = possible[0]
            print (on_cpu.get_unique_name(), " uses the processor. ")
            html += '<div class="wrapper"><div class="counter">'+ str(i) +'~'+ str(i+1) +' </div><div class="cpu_used" data-toggle="modal" data-target="#myModal">' + on_cpu.get_unique_name() + '</div></div>'
            cpu_tasks.append({'cpu_time':i,'task_instance':on_cpu})
            task_timeline.append({'cpu_time':i,'task_instance':on_cpu})
            if on_cpu.use( clock_step ):
                tasks.remove( on_cpu )
                print ("Finish!")
        else:
            print ('No task uses the processor. ')
            task_timeline.append( {'cpu_time': i, 'task_instance': None} )
            html += '<div class="wrapper"><div class="counter">'+ str(i) +'~'+ str(i+1) +' </div><div class="cpu_empty">Empty</div></div>'
            print ("\n")
    #Print remaining periodic tasks
    html += "<br /><br />"
    s1 = on_cpu.name
    for t in task_types:
        s2 = t.name.rstrip()
        if s1 == s2:
            ET = (i) - on_cpu.start
            if t.RT > ET or t.RT == -1:
                t.RT = ET

            if i <= on_cpu.priority_deadline:
                t.times += 1
            else:
                t.missed += 1

    for p in tasks:
        print (p.get_unique_name() + " is dropped due to overload!")
        html += "<p>" + p.get_unique_name() + " is dropped due to overload!</p>"





    return render_template("fef.html", **locals())


def _getUrgencyFromClass(priority_string):
    if(priority_string == "Normal Periodic"):
        return 0
    if (priority_string == "High Priority Periodic"):
        return 1
    if (priority_string == "Normal Event Driven"):
        return 2
    if (priority_string == "High Urgency Event Driven"):
        return 3
    return -1

def _prepareTaskInstances(hyperperiod, task_types, tasks, total):
    instance_data = []

    file_headers = ['Instance Id','Tasks No', 'Start', 'End', 'Priority', 'Task Name', 'Urgency', 'Priority Deadline', 'Priority Period',
                    'Priority Slack', 'Priority Execution',  'Sensing Output']
    instance_data.append(file_headers)

    file = open("instances.csv", 'w', newline='')
    writer = csv.writer(file, delimiter=',', quotechar='"')

    for i in np.arange(0, hyperperiod):
        for task_type in task_types:
            # check if its event driven tasks
            vo_name = task_type.vo
            if task_type.period == 0 and i == task_type.release:

                Task_no, end, pD, pE, pP, pS, priority, start, urgency, i_output = _createTaskInstances(i, task_type)

                task_instance = TaskIns(start=start, end=end, priority=priority, name=task_type.name, urgency=urgency, pD=pD,
                            pP=pP, pS=pS, pE=pE, Tno=Task_no, deadline=task_type.deadline, i_out=i_output,pB=0,vo=vo_name)
                tasks.append(task_instance)
                instance_data.append([str(task_instance.id), str(task_instance.Tno), str(task_instance.start),
                                     str(task_instance.end),
                                     str(task_instance.priority), str(task_instance.name), str(task_instance.urgency),
                                     str(task_instance.priority_deadline),
                                     str(task_instance.priority_period), str(task_instance.priority_slack), str(task_instance.priority_exec),
                                     str(task_instance.i_out)])
                total = total + task_type.execution
            # In case tasks are periodic
            elif task_type.period != 0:
                if (i - task_type.release) % task_type.period == 0 and i >= task_type.release:
                    Task_no, end, pD, pE, pP, pS, priority, start, urgency, i_output = _createTaskInstances(i, task_type)

                    task_instance = TaskIns(start=start, end=end, priority=priority, name=task_type.name, urgency=urgency, pD=pD,
                                pP=pP, pS=pS, pE=pE, deadline=task_type.deadline, Tno=Task_no,i_out=i_output,pB=0, vo=vo_name)
                    tasks.append(task_instance)
                    instance_data.append([str(task_instance.id), str(task_instance.Tno), str(task_instance.start), str(task_instance.end),
                                 str(task_instance.priority), str(task_instance.name), str(task_instance.urgency), str(task_instance.priority_deadline),
                                         str(task_instance.priority_period), str(task_instance.priority_slack),
                                         str(task_instance.priority_exec), str(task_instance.i_out)])
                    #print(event_instance.start, event_instance.id)

                    total = total + task_type.execution

    print(instance_data)
    writer.writerows(instance_data)
    instance_data = []
    file.close()
    return total


def _createTaskInstances(i, task_type):
    start = i
    end = start + task_type.execution
    priority = start + task_type.deadline
    urgency = task_type.urgency
    Task_no = task_type.Tno
    pD = start + task_type.deadline
    pP = task_type.period
    pS = (start + task_type.deadline) - (start + task_type.execution)
    pE = task_type.execution
    i_out =  random.randint(10, 100)
    return Task_no, end, pD, pE, pP, pS, priority, start, urgency, i_out



@app.route('/fef')
def fef():
    # Variables
    # html_color = { 'Task1':'red', 'Task2':'blue', 'Task3':'green', 'Task4':'aqua', 'Task5':'coral', 'Empty':'grey', 'Finish':'black'}
    global scenarios
    global sampling
    filename = request.args.get("filename")
    page_title = "Priority Based Hybrid Scheduling"
    html_color = {'Core1': 'red', 'Core2': 'blue', 'Core3': 'green', 'Core4': 'aqua', 'Empty': 'grey',
                  'Finish': 'black'}
    scenario_id = request.args.get('scenario_id')
    task_types = []
    tasks = []
    hyperperiod = []
    No = 0

    hyperperiod, task_types, No = _tasksReaderDB()



   # package_dir = os.path.dirname(os.path.abspath(__file__) + "/files/")
   # thefile = os.path.join(package_dir, filename)


    #print ("Hyper period: " + str( hyperperiod ))


    total = 0
    # Create task instances

    # This function goes through all the task typers and create task instances and return total instances.
    total = _prepareTaskInstances(hyperperiod, task_types, tasks, total)

    # Suggest No of Procesors
    # print "HP: " + str(HP)
    # print "Required: " +str(total)
    core = round( total/hyperperiod )
    print ("Least No of Cores Needed: " + str( core ))

    # Html output start
    html = "<!DOCTYPE html><html><head><title>EDF Scheduling</title></head><body>"

    # Simulate clock
    clock_step = 1
    res = 1
    PF = 0  # processor utilization
    MissedDeadline = []
    task_timeline = []
    cpu_tasks = []

    for i in np.arange( 0, hyperperiod, clock_step ):  # hyperperiod -> 6
        # Fetch possible tasks that can use cpu and sort by priority
        possible = []
        Periodic = []
        Periodic_Period = []
        Periodic_Deadline = []
        Periodic_Slack = []
        Event = []
        Event_Urgency = []
        Event_Deadline = []
        Event_Slack = []
        FM = 0

        Event_SameUrgenecy = []
        Event_SameDeadline = []

        MightSafe = []

        for t in tasks:
            if t.start <= i:
                possible.append( t )
                if t.urgency > 1:
                    if int(float(t.urgency)) == 2:
                        Event.append( t )
                    else:
                        Event_Urgency.append( t )
                else:
                    Periodic.append( t )


        possible = sorted( possible, key=cmp_to_key(priority_cmp) )
        Periodic = sorted( Periodic, key=cmp_to_key(priority_cmp) )
        Periodic_Deadline = sorted( Periodic, key=cmp_to_key(priority_cmp_deadline) )
        Periodic_Period = sorted( Periodic, key=cmp_to_key(priority_cmp_period) )
        Periodic_Slack = sorted( Periodic, key=cmp_to_key(priority_cmp_slack) )

        Event_Deadline = sorted( Event, key=cmp_to_key(priority_cmp_deadline) )
        Event_Urgency = sorted( Event_Urgency, key=cmp_to_key(priority_cmp_Urgency) )
        Event_Slack = sorted( Event, key=cmp_to_key(priority_cmp_slack) )

        if i == 0:
            print ("\n")
            print ("Arrived Tasks at time: " + str( i ))

            print ("Event_Urgency: ")
            for j3 in range( 0, len( Event_Urgency ) ):
                print ("    " + str( Event_Urgency[j3] ))

            print ("Event: ")
            for j1 in range( 0, len( Event ) ):
                print ("    " + str( Event[j1] ))

            print ("Periodic: ")
            for j2 in range( 0, len( Periodic ) ):
                print ("    " + str( Periodic[j2] ))
            print ("\n")

        NoCores = 1

        for j in range( 0, NoCores ):
            # Select task with highest priority

            if len( possible ) > 0:
                if len( Event ) > 0 or len( Event_Urgency ) > 0:
                    if len( Event_Urgency ) > 0:
                        if len( Event_Urgency ) == 1:
                            on_cpu = Event_Urgency[0]
                        else:
                            TempDeadline = []
                            TempDeadline = sorted( Event_Urgency, key=cmp_to_key(priority_cmp_deadline) )
                            Event_SameDeadline.append( TempDeadline[0] )
                            for k in range( 1, len( TempDeadline ) ):
                                if (TempDeadline[0].priority_deadline == TempDeadline[k].priority_deadline):
                                    Event_SameDeadline.append( TempDeadline[k] )
                            if len( Event_SameDeadline ) > 1:
                                TempArrival = []
                                TempArrival = sorted( Event_SameDeadline, key=cmp_to_key(priority_cmp_arrival) )
                                on_cpu = TempArrival[0]
                            else:
                                on_cpu = TempDeadline[0]
                    else:
                        if len( Periodic ) > 0:  #########################################
                            if (Event_Slack[0].priority_slack >= (2 * (Periodic_Slack[0].priority_slack))):  ##
                                on_cpu = Periodic_Slack[0]  ##       Urgency Calculation;
                            else:  ## threshold can be changed with survey
                                on_cpu = Event_Slack[0]  ##

                elif len( Event_Urgency ) == 0:
                    if len( Periodic ) > 0:
                        Periodic_Deadline = sorted( Periodic, key=cmp_to_key(priority_cmp_deadline) )
                        Periodic_Period = sorted( Periodic, key=cmp_to_key(priority_cmp_period) )
                        Periodic_Slack = sorted( Periodic, key=cmp_to_key(priority_cmp_slack) )
                        MightMiss = []
                        Periodic_SameDeadline = []
                        Periodic_SameSlack = []
                        Periodic_SamePeriod = []

                        m = len( Periodic_Deadline )
                        FD = Periodic_Deadline[m - 1].priority_deadline

                        for k1 in range( 0, m ):
                            if (Periodic[k1].priority_slack < 1):
                                MightMiss.append( Periodic[k1] )
                            else:
                                MightSafe.append( Periodic[k1] )

                        TotalExec = i

                        for t in tasks:
                            if (t.end - 3) < FD:  # -3 added to consider deadlines at the edge
                                TotalExec = TotalExec + t.priority_exec

                        if (TotalExec > FD):
                            FM = 1

                        if FM == 0 and len( MightMiss ) > 0:
                            print ("FD: " + str( FD ))
                            print ("TotalExec: " + str( TotalExec ))
                            if (len( MightMiss ) == 0):
                                on_cpu = MightMiss[0]
                            else:
                                MightMiss = sorted( MightMiss, key=cmp_to_key(priority_cmp_slack) )
                                Periodic_SameSlack.append( MightMiss[0] )
                                for k3 in range( 1, len( MightMiss ) ):
                                    if (MightMiss[0].priority_slack == MightMiss[k3].priority_slack):
                                        Periodic_SameSlack.append( MightMiss[k3] )
                                if len( Periodic_SameSlack ) > 1:
                                    TempPeriod = []
                                    TempPeriod = sorted( Periodic_SameSlack, key=cmp_to_key(priority_cmp_period ))
                                    on_cpu = TempPeriod[0]
                                else:
                                    on_cpu = MightMiss[0]
                        else:
                            if len( Periodic_Period ) > 1:
                                Periodic_SamePeriod.append( Periodic_Period[0] )
                                for k4 in range( 1, len( Periodic_Period ) - 1 ):
                                    if (Periodic_Period[0].priority_period == Periodic_Period[k4].priority_period):
                                        Periodic_SamePeriod.append( Periodic_Period[k4] )
                                if len( Periodic_SamePeriod ) > 1:
                                    TempArrival = []
                                    TempArrival = sorted( Periodic_SamePeriod, key= cmp_to_key(priority_cmp_arrival) )
                                    on_cpu = TempArrival[0]
                                else:
                                    on_cpu = Periodic_Period[0]
                            else:
                                on_cpu = Periodic_Period[0]


                on_cpu.priority_exec -= 1
                for t in tasks:
                    if t.start <= i:
                        if t.priority_slack != 0:
                            t.priority_slack -= 1



                #print(on_cpu.deadline, on_cpu.urgency)
                if(on_cpu.priority_period == 0 and on_cpu.urgency == 1):
                    cls = "urgent-event"
                if (on_cpu.priority_period == 0 and on_cpu.urgency == 0):
                    cls = "normal-event"
                if (on_cpu.priority_period != 0 and on_cpu.urgency == 1):
                    cls = "urgent-periodic"
                if (on_cpu.priority_period != 0 and on_cpu.urgency == 0):
                    cls = "normal-periodic"

                html += '<div class="wrapper"><div class="counter">'+ str(i) +'~'+ str(i+1) +' </div><div class="cpu_used '+cls+'">' + on_cpu.get_unique_name()  +  '</div>'
                cpu_tasks.append( {'cpu_time': i, 'task_instance': on_cpu} )
                task_timeline.append( {'cpu_time': i, 'task_instance': on_cpu} )

                if on_cpu.use( clock_step ):
                    tasks.remove( on_cpu )
                    possible.remove( on_cpu )
                    if on_cpu.urgency > 1:
                        if on_cpu.urgency == 2:
                            Event.remove( on_cpu )
                        else:
                            Event_Urgency.remove( on_cpu )
                    else:
                        Periodic.remove( on_cpu )

                    s1 = on_cpu.name
                    for t in task_types:
                        s2 = t.name.rstrip()
                        if s1 == s2:
                            ET = (i) - on_cpu.start
                            if t.RT > ET or t.RT == -1:
                                t.RT = ET

                            if i <= on_cpu.priority_deadline:
                                t.times += 1
                            else:
                                t.missed += 1

                html += '</div>'

                if i > on_cpu.priority_deadline:
                    MissedDeadline.append( on_cpu )
                    # print "...... Missed Deadline...."
                    # print on_cpu


            else:
                print ('No task uses the processor. ')
                html += '<div class="wrapper"><div class="counter">'+ str(i) +'~'+ str(i+1) +' </div><div class="cpu_empty">Empty</div></div>'
                task_timeline.append( {'cpu_time': i, 'task_instance': None} )
                PF = PF + 1;  # processor free
                # print "\n"
    count = 0
    missedD = 0
    # Print remaining periodic tasks
    html += "<br /><br />"
    for p in tasks:
        if p.priority <= hyperperiod:
            # print p.get_unique_name() + " is dropped due to overload!  " + str(p.start) +" "+ str(p.priority)+" "+str(p.end)
            count = count + 1

    if len( MissedDeadline ) > 0:
        missedD = len( MissedDeadline )
        print ("Missed Deadlines = " + str( missedD ))

    print ("Tasks Missed: " + str( count ))
    print ("\n")

    # Print Task Schedulability
    html += "<br /><br />"
    if count == 0:
        print ("Task is Schedulable")
        html += "<p> Task is Schedulabl!</p>"
    else:
        print ("Task is NOT Schedulable")
        html += "<p> Task is NOT Schedulabl!</p>"

    print ("\n")

    # Print Processor Utilization
    html += "<br />"

    if PF == 0:
        print ("Processor was free " + str( PF ) + " units of time")
        print ("Processor Utlization = 100%")
        html += "Processor was free " + str( PF ) + " units of time <br />"
        html += "Processor Utlization = 100%"
    else:
        print ("Processor was free " + str( PF ) + " units of time \n")
        html += "Processor was free " + str( PF ) + " units of time <br />"

    print ("\n")

    # No of times ech task ran
    for t in task_types:
        print (str( t.name.rstrip() ) + " completed " + str( t.times ) + " times during the hyper-period")
        html += "<p>" + t.name + " completed " + str( t.times ) + " times during the hyper-period</p>"
    for t in task_types:
        print (str(t.name.rstrip()) + " missed " + str(t.missed) + " times during the hyper-period")
        html += "<p>" + t.name + " missed " + str(t.missed) + " times during the hyper-period</p>"
    task_labels = []
    task_rt = []
    #icolor = palette.Color("#aaaa00")
    #bg_color = ["#aaaa00"]
    for t in task_types:
        print (str(t.Tno) + " response time: " + str(t.RT))
        #response_time.append({'id':t.Tno, 'rt':t.RT})
        task_labels.append(t.name)
        task_rt.append(t.RT)
        #icolor.rgb8.r = (icolor.rgb8.r + 10) % 255
        #icolor.rgb8.g = (icolor.rgb8.g -10) % 255
        #icolor.rgb8.b = (icolor.rgb8.b + 10) % 255

        #bg_color.append(icolor.hex)

        html += "<p>" + t.name + " response time: " + str(t.RT) + " </p>"

        task_types = sorted(task_types, key=getKey)

    # Html output end
    html += "</body></html>"

    return render_template("fef.html", **locals())

@app.route('/help')
def help():
    return render_template("help.html")

#################### Utility Functions######################################
def  _tasksReader(fileName):
    if os.path.isfile(fileName) == 0:
        return 0

    file =  open( fileName, 'rt' )
    task_types = []
    hyperperiod = []
    try:
        reader = csv.reader(file)
        i=0;
        total_tasks = 0
        for row in reader:
            if(i > 0):
            # temp_p = int(row(1))
            # if(temp_p > 0):
            #     hyperperiod.append(temp_p)

                if(int(row[1]) > 0):
                    hyperperiod.append(int(row[1]))
                task_types.append(TaskType(period=int(row[1]), release=int(row[3]), execution=int(row[2]), deadline=int(row[4]), name='task'+row[0], time=0,
                                           Tno=int(row[0]), missed=0, urgency=row[5]))

            i = i+1
            total_tasks = total_tasks + 1
    finally:

        hyperperiod = lcm(hyperperiod)
        task_types = sorted( task_types, key=cmp_to_key( tasktypes_cmp ) )
        file.close()
    return hyperperiod,task_types, total_tasks
############################################################################

#################### Utility Functions######################################
def  _tasksReaderDB():
    task_types = []
    hyperperiod = []
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT mt.task_id, mt.vo_id, t.title, t.priority, t.arrival, t.period, t.execution, t.deadline, v.name FROM `mapped_task` mt INNER JOIN tasks t ON mt.task_id = t.id INNER JOIN virtualobjects v ON mt.vo_id = v.id ")
        tasks = cur.fetchall()

        i=0
        total_tasks = 0
        for row in tasks:
            if(int(row['period']) > 0):
                hyperperiod.append(int(row['period']))
            task_types.append(TaskType(period=int(row['period']), release=int(row['arrival']), execution=int(row['execution']),
                                       deadline=int(row['deadline']), name=row['title'], time=0,
                                       Tno=int(row['task_id']), vo=row['name'], missed=0, urgency=(row['priority'])))
            total_tasks = total_tasks + 1
    finally:

        hyperperiod = lcm(hyperperiod)
        task_types = sorted( task_types, key=cmp_to_key( tasktypes_cmp ) )

    return hyperperiod,task_types, total_tasks

if(__name__ == '__main__'):
    app.run(debug=True)