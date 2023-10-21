"""
==========================================
Fuzzy Control Systems: Trip Grading
==========================================
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
NumPy at least 1.23.4
Matplotlib at least 3.6.2
Scikit-fuzzy 0.4.2
Link to install python: https://www.python.org/downloads/
When you have installed python in the command line run commands
pip install scikit-fuzzy
pip install numpy
pip install matplotlib

==========================================
Trips grading problem.
==========================================

We created a fuzzy logic system to grade trips with guides on percentage levels.
When grading the trip, you consider the guide skills and professionalism, route attractiveness, price of the trip and
logistics of the trip. All of them rated between 1 and 5(classic full stars model known from websites). We assume
the idea that not all of these elements of grades has this same weight, so we are proposing fuzzy logic system to
estimate overall grade for a trip. Trip will be graded between 0% and 100% satisfaction.

We would formulate this problem as:

* Antecednets (Inputs)
   - `guide professionalism`
      * Universe: How good was the guide, on a scale of 1 to 5?
      * Fuzzy set: poor, mediocre, average, decent, good
   - `route attractiveness`
      * Universe: How tasty was the food, on a scale of 0 to 10?
      * Fuzzy set: poor, mediocre, average, decent, good
* Consequents (Outputs)
   - `tip`
      * Universe: How much should we tip, on a scale of 0% to 25%
      * Fuzzy set: low, medium, high
* Rules
   - IF the *service* was good  *or* the *food quality* was good,
     THEN the tip will be high.
   - IF the *service* was average, THEN the tip will be medium.
   - IF the *service* was poor *and* the *food quality* was poor
     THEN the tip will be low.
* Usage
   - If I tell this controller that I rated:
      * the service as 9.8, and
      * the quality as 6.5,
   - it would recommend I leave:
      * a 20.2% tip.

"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


"""
Antecedent of guide professionalism, route attractiveness, price and logistics, all between 1 and 5
Consequent of trip grade between 0 and 100%
"""
guide_professionalism = ctrl.Antecedent(np.arange(1, 6, 1), 'guide_professionalism')
route_attractiveness = ctrl.Antecedent(np.arange(1, 6, 1), 'route_attractiveness')
price_quality = ctrl.Antecedent(np.arange(1, 6, 1), 'price_quality')
logistics_aspects = ctrl.Antecedent(np.arange(1, 6, 1), 'logistics_aspects')
trip_grade = ctrl.Consequent(np.arange(0, 101, 1), 'trip_grade')

# Auto-membership function for 5 levels per input
guide_professionalism.automf(5)
route_attractiveness.automf(5)
price_quality.automf(5)
logistics_aspects.automf(5)

# Custom membership functions for trip grade output
trip_grade['very low'] = fuzz.trimf(trip_grade.universe, [0, 0, 20])
trip_grade['low'] = fuzz.trimf(trip_grade.universe, [0, 20, 40])
trip_grade['medium'] = fuzz.trimf(trip_grade.universe, [20, 40, 60])
trip_grade['high'] = fuzz.trimf(trip_grade.universe, [40, 60, 80])
trip_grade['very high'] = fuzz.trimf(trip_grade.universe, [60, 80, 100])

# If you want to observe your membership functions you can uncomment these lines
# guide_professionalism.view()
# route_attractiveness.view()
# trip_grade.view()
"""
Fuzzy rules
-----------

Now, to make these triangles useful, we define the *fuzzy relationship*
between input and output variables. For the purposes of our example, consider
three simple rules:

1. If the food is poor OR the service is poor, then the tip will be low
2. If the service is average, then the tip will be medium
3. If the food is good OR the service is good, then the tip will be high.

Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""
# guide_professionalism, route_attractiveness, price_quality, logistics_aspects, trip_grade
rule1 = ctrl.Rule(guide_professionalism['poor'] |
                  route_attractiveness['poor'] |
                  price_quality['poor'] |
                  logistics_aspects['poor'], trip_grade['very low'])
rule2 = ctrl.Rule(guide_professionalism['mediocre'] |
                  route_attractiveness['mediocre'] |
                  price_quality['mediocre'] |
                  logistics_aspects['mediocre'], trip_grade['low'])
rule3 = ctrl.Rule(guide_professionalism['average'] |
                  route_attractiveness['average'] |
                  price_quality['average'] |
                  logistics_aspects['average'], trip_grade['medium'])
rule4 = ctrl.Rule(guide_professionalism['decent'] |
                  route_attractiveness['decent'] |
                  price_quality['decent'] |
                  logistics_aspects['decent'], trip_grade['high'])
rule5 = ctrl.Rule(guide_professionalism['good'] |
                  route_attractiveness['good'] |
                  price_quality['good'] |
                  logistics_aspects['good'], trip_grade['very high'])
rule6 = ctrl.Rule(guide_professionalism['average'] |
                  route_attractiveness['average'] |
                  price_quality['poor'] |
                  logistics_aspects['poor'], trip_grade['low'])
rule7 = ctrl.Rule(guide_professionalism['good'] |
                  route_attractiveness['good'] |
                  price_quality['mediocre'] |
                  logistics_aspects['mediocre'], trip_grade['high'])
rule8 = ctrl.Rule(guide_professionalism['poor'] |
                  route_attractiveness['poor'] |
                  price_quality['good'] |
                  logistics_aspects['good'], trip_grade['low'])


"""
Now that we have our rules defined, we can simply create a control system
via:
"""
trip_grade_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])

"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``. Think of this object representing our controller
applied to a specific set of cirucmstances. For grading trips, this might be grading
Stefan's trip on the old town. We would create another
``ControlSystemSimulation`` when we're trying to apply our ``tipping_ctrl``
for Jurek's trip in Kashubian's forests.
"""
trip_grading = ctrl.ControlSystemSimulation(trip_grade_control)

"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the guide professionalism 1 out of 5,
route attractiveness 1 of 5, price 5 out of 5, and logistics 5 out of 5.

Pass inputs to the ControlSystem using Antecedent labels in a form of dictionary, 
where key is name of the input and value is a number of stars of specific trip aspect
"""
data_dict = {'guide_professionalism': 1, 'route_attractiveness': 1, 'price_quality': 5, 'logistics_aspects': 5}
trip_grading.inputs(data_dict)

# Crunch the numbers
trip_grading.compute()

"""
Once computed, we can view the result as well as visualize it.
"""
print(trip_grading.output['trip_grade'])
trip_grade.view(sim=trip_grading)

"""
The resulting suggested grade is **44.81%**.
"""
# Open the diagram of our systems results
plt.show()
