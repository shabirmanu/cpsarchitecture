from flask import Flask
from enum import EnumMeta, Enum
from flask_wtf import Form
from wtforms import IntegerField, BooleanField, StringField, TextAreaField, SelectField
from wtforms import validators, ValidationError, FileField


class GenerateTasks(Form):
    noOfTasks = IntegerField('No of Tasks', [validators.required("Please enter number of tasks to be generated")])
    periodRangeFrom = IntegerField('Period Range: (From)', [validators.required("Please enter Period start")])
    periodRangeTo = IntegerField('Period Range (To)', [validators.required("Please enter Period End")])
    timeRangeFrom = IntegerField('Start Time Range (From)',[validators.required("Please enter Arrival Time Start")] )
    timeRangeTo = IntegerField('Start Time Range (To)', [validators.required("Please enter Arrival time End")])
    execRangeFrom = IntegerField('Execution Time Range (From)', [validators.required("Please enter Execution time start")])
    execRangeTo = IntegerField('Execution Time Range (To)', [validators.required("Please enter Exectution time End")])




class AddVirtualObject(Form):
    name = StringField('Name', [validators.required("Please enter title for the task.")])
    taskTags = StringField('Tags', [validators.required("Please enter tags.")])
    url = StringField('Endpoint URL', [validators.required("Please enter the URL.")])
    methods = StringField('Methods', [validators.required("Please enter Methods of Virtual Objects")])
    attributes = StringField('Properties', [validators.required("Please enter Properties of Virtual Object")])

class AddService(Form):
    title = StringField('Name', [validators.required("Please enter title for the service.")])
    description = TextAreaField('Description', [validators.required("Please enter Description for the service.")])

class AddProject(Form):
    name = StringField('Name', [validators.required("Please enter descriptive name for the project.")])
    description = TextAreaField('Description', [validators.required("Please enter Description for the project.")])



class AddSolverType(Form):
    name = StringField('Name', [validators.required("Please enter descriptive name for the project.")])
    hyperparameters = TextAreaField('Hyper parameters', [validators.required("Please enter Description for the project.")])
    algorithm = SelectField(
        "Algorithm",
        choices=[('','Select Algorithm'),('Prediction','Prediction'), ('Optimization','Optimization'), ('Scheduling','Scheduling'), ('Control','Control')])

class AddObjectiveFunction(Form):
    name = StringField('Name', [validators.required("Please enter descriptive name for the project.")])
    # Model Declaration and Assignment
    des_param = TextAreaField('Design Parameters Declaration', [validators.required("Please declare Design parameter.")])
    # Data Assignments
    obf = TextAreaField('Cost', [validators.required("Please enter Description for the project.")])
    constraints = TextAreaField('Constraints')
    data_assign = TextAreaField('Data Assignment')
    goal = SelectField(
        "Goal",
        choices=[('0', 'Minimize'), ('1', 'Maximize')])
    dataset = FileField('Dataset (Optional)')

class AddPredictionObjectiveForm(Form):
    name = StringField('Name', [validators.required("Please enter descriptive name for the project.")])
    in_features = TextAreaField('Input Features', [validators.required("Please enter input features.")])
    op_features = StringField('Output Features', [validators.required("Please enter output feature along with dataset position.")])
    hyperparameter = TextAreaField('Hyperparameter')
    dataset = FileField('Dataset')