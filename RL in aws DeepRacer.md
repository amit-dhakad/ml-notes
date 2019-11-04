                													AWS DeepRacer



#RL in aws DeepRacer

## Recap

AWS DeepRacer:

- is fully-autonomous
- is a 1/18th scale racing car
- utilizes RL to learn driving habits

Types of machine learning include:

- Supervised learning
- Unsupervised learning
- Reinforcement learning





## Unboxing Recap

AWS DeepRacer includes the following in its box:

1. Car chassis (with compute and camera modules)
2. Car body
3. Car battery
4. Car battery charger
5. Car battery power adapter
6. Compute module power bank
7. Power bank’s connector cable
8. Power adapter and power cord for compute module and power bank
9. 4 pins (car chassis pins)
10. 12 pins (spare parts)
11. USB-A to micro USB cable





## Under the Hood Recap

Other than the vehicle chassis, the vehicle also includes:

- a HD DeepLens camera
- expansion ports
- HDMI port
- microUSB port
- USB-C port
- on/off switch
- reset button
- LEDs

## Steering Calibration Recap

To calibrate your steering:

1. Place your car on a block (secure it with duct tape or similar) to keep it in place while the wheels move.
2. Get your vehicle’s IP address from when it was [set up to use wi-fi](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-set-up-vehicle.html#deepracer-set-up-vehicle-wifi-connection).
3. Sign in to the AWS DeepRacer console with that IP address as instructed [here](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-set-up-vehicle.html#deepracer-set-up-vehicle-test-drive).
4. Select **Calibration**, then **Calibrate Steering Angle** from the console (see [here](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-calibrate-vehicle.html)).
5. Calibrate your center steering first. *Remember:* Due to the concepts of [Ackermann steering](http://datagenetics.com/blog/december12016/index.html), only one wheel will actually be straight when you calibrate center steering.
6. Calibrate your left steering by choosing the value where your vehicle wheels will turn no further left.
7. Calibrate your right steering by choosing the value where your vehicle wheels will turn no further right.

## Throttle Calibration Recap

To calibrate your throttle (you have already done steps 1-3 if you just came from Steering Calibration):

1. Place your car on a block (secure it with duct tape or similar) to keep it in place while the wheels move.
2. Get your vehicle’s IP address from when it was [set up to use wi-fi](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-set-up-vehicle.html#deepracer-set-up-vehicle-wifi-connection).
3. Sign in to the AWS DeepRacer console with that IP address as instructed [here](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-set-up-vehicle.html#deepracer-set-up-vehicle-test-drive).
4. Select **Calibration**, then **Calibrate Speed** from the console (see [here](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-calibrate-vehicle.html)).
5. Calibrate your stopped speed by moving the bar until the wheels are no longer moving.
6. Calibrate the forward motion by moving the bar and checking the direction the wheels turn. If they turn clockwise, the forward direction is set. If not, select the “Reverse direction” button to switch the direction the wheels turn to go forward.
7. Calibrate the forward maximum speed. Typically, it’s best not to go too high above 2 m/s, as that is the highest the simulator provides in its action space.
8. Calibrate the maximum backward speed, which should be essentially the negative value of what the maximum forward speed was set at.







# AWS DeepRacer Workflow



In just a minute, you’ll have a chance to get into the AWS DeepRacer Console and build your first model. But before doing that, let’s walk through the workflow behind AWS DeepRacer. Throughout this course, you’ll get to dig in deeper to all of these separate steps, but for now, just take a minute to understand what’s happening at a more superficial level.

Whether you end up racing on a virtual or physical track, you’ll want to start in the simulated environment – building, training, evaluating, and repeating that process until you’ve optimized your model enough so that you think it’s ready to compete. Then, if you’d like, you can deploy to the actual physical car.



[![AWS DeepRacer Workflow](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d116eb6_l2-deepracer-workflow/l2-deepracer-workflow.png)AWS DeepRacer Workflow](https://classroom.udacity.com/courses/ud012/lessons/1a8116db-d6ca-4685-81ae-666b03bcb0f9/concepts/3b9c4ea4-9816-4bc4-858a-60761fdcb300#)



This all starts with creating a model in the AWS DeepRacer console. You use the console for model configuration. Once built, the simulator is used to train and then evaluate your model. The simulator is a great tool: you can see in *real time* how your model is responding to the settings you’ve chosen.

After your model is trained, you can evaluate its performance in the simulator, which yields the percentage of the lap complete for each of your runs, plus the overall time it took to complete each run.

If you like what you’ve done with your trained and evaluated model, you can clone it and start to tweak your configurations for increased performance.





#The console and first model

## Recap

We'll see these again in the exercise coming soon - so you don't need to work through it just yet!

1. Sign into [the console](http://console.aws.amazon.com/deepracer/home)
2. Click the Get Started / Create Model button on the right
3. Create resources, if they aren't created yet
4. Name and describe your model
5. Choose an environment (`re:Invent 2018` for now)
6. Select action space
7. Select reward function
8. Tune hyperparameters
9. Set stop condition





# The Basic Reward Function

We briefly covered the reward function in the last video, but let’s take another look at how our basic reward function behaves.



[![The basic reward function operates off distance from the center of the lane, using “markers” at different percentages of track width.](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d116efd_l2-reward-function/l2-reward-function.png)The basic reward function operates off distance from the center of the lane, using “markers” at different percentages of track width.](https://classroom.udacity.com/courses/ud012/lessons/1a8116db-d6ca-4685-81ae-666b03bcb0f9/concepts/96e31c48-085e-4762-a3ee-dd22059b1e2f#)



In the above image, you can see that there are different rewards based on how far the car is from the center line. Here is the code from the AWS console itself:

```python
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    return float(reward)
```

In the above code, we first gather the parameters for track width and the distance from the center line (we’ll discuss all the parameters available to you in more detail later). Then, we create three markers based off of the track width - one at 10% of the track width, another at 25% width, and the last at half of the track width.

From there, it’s a simple if/else statement that gives different, decreasing rewards based on the vehicle being within a given marker or not. If it is outside all three markers, notice that the reward is almost effectively zero.





# Exercise I: Model Training Using the AWS DeepRacer Console



This exercise walks you through building, training, and evaluating your first reinforcement learning model via the AWS DeepRacer console. You can follow along with the [Resources Pack](https://s3.amazonaws.com/awsu-hosting/DIG-TF-200-MLDRRL-10-EN/1.0/resource-pack.zip) from AWS, although we also include the instructions here in the classroom.

*Note:* This exercise is designed to be completed in your AWS account. AWS DeepRacer is part of AWS Free Tier, so you can get started with the service at no cost. For the first month after sign-up, you are offered a monthly free tier of 10 hours of Amazon SageMaker training, and 60 simulation units of Amazon RoboMaker (enough to cover 10 hours of training). For more information, see the [AWS DeepRacer Pricing page](https://aws.amazon.com/deepracer/pricing/).



[![AWS RoboMaker is one of the platforms used by AWS DeepRacer - in this case, for the simulated track!](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d116f1f_l2-aws-robomaker/l2-aws-robomaker.png)AWS RoboMaker is one of the platforms used by AWS DeepRacer - in this case, for the simulated track!](https://classroom.udacity.com/courses/ud012/lessons/1a8116db-d6ca-4685-81ae-666b03bcb0f9/concepts/c5d82715-3101-4d38-99d1-f0a7008a4dbf#)



## Learning Objectives

By the end of this exercise, you will be able to:

1. Access and navigate the AWS DeepRacer console
2. Identify the steps that go into building a model in the AWS DeepRacer console
3. Use the basic reward function in AWS DeepRacer when configuring a model
4. Use the AWS DeepRacer simulator to train and evaluate a model

## Technical Prerequisites

- Experience using AWS technologies
- Basic understanding of machine learning concepts, particularly reinforcement learning and how it applies to AWS DeepRacer





# Exercise I: Task I



[![An example from the training console.](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d116f58_l2-deepracer-training/l2-deepracer-training.png)An example from the training console.](https://classroom.udacity.com/courses/ud012/lessons/1a8116db-d6ca-4685-81ae-666b03bcb0f9/concepts/4670ac96-2ee4-496d-bdeb-7906928e82d7#)



## Task 1: Create your first AWS DeepRacer model

This exercise is made up of one main task that has you create, train, and evaluate a model using the basic reward function and other default settings within the AWS DeepRacer console. You will have a chance to build on this model in order to enhance your car’s performance in later exercises. To get started building your first model, follow the steps below.

1. In your AWS account, go to the Management Console. (You can also directly sign in to the AWS DeepRacer Console in the upper-right [here](https://aws.amazon.com/deepracer/), which skips the next couple of steps).
2. Choose the `us-east-1` (N. Virginia) region at the top right corner of the **Regions** dropdown menu.
3. From the top left of the console, click **Services**, type **DeepRacer** in the search box, and select **AWS DeepRacer**. That will open the AWS DeepRacer console.
4. On the landing page, if it is your first time visiting AWS DeepRacer, you may need to select **Get Started**, and then **Create Resources** to provision the necessary AWS resources. Otherwise, you can click **Create model**.
5. Create a model name and description.
6. Under **Environment simulation**, select a track from the list. It’s recommended to start with the “re:Invent 2018 Track” and then explore more tracks from there.
7. Under **Action space**, familiarize yourself with these settings and accept the defaults.
8. Read through the default (basic) reward function and check out the other examples. Note that you will want to “validate” if you update the default in the future.
9. Expand **Hyperparameters** to review the different hyperparameters available to set. For this exercise, accept the default hyperparameter settings.
10. For the Stop conditions go ahead and choose a max time of 60 minutes and click **Start training**.
11. The training section will allow you to see the reward function over time, as well as see a live stream of the simulator. Note that you may need to re-load the page if the stream does not update. Select the name of the model to watch the live stream in the simulator. Notice how your car is moving and become familiar with the general look and feel of the simulation. *The new training will initiate in about 6 minutes.* You then must wait for the training job to complete before proceeding. If you choose a maximum time of 60 minutes, it will take up to 60 minutes for this training job to complete. It is complete when the status reads “Ready.” **Note:** *Since the training job may take more than an hour, you might want to consider moving on to the next lesson of this course while the training job is running.*
12. Now, select the name of the model you just trained and click **Start evaluation** (*Note: You can also click on **Reinforcement Learning** on the left to get either your trained model or some pre-trained sample models to evaluate. Click on the name, then **Start new evaluation**).
13. Select the same track you used for training.
14. Set the number of trials to **3** and click **Start evaluation**. *The evaluation results will update in approximately 5 minutes.*
15. Before the evaluation job completes, take some time to watch the simulator to see how your car’s performing in real time. Look for things that you might want to tweak in the future and note them down. You’ll have a chance to apply some of those ideas in a later exercise in the course.
16. Watch the evaluation results update and write them down. You will use your results from this model as a benchmark that you can compare to as you tweak your model in later exercises.







# RL and DeepRacer

Here, we'll review reinforcement learning to give you a conceptual understanding of what's happening under the hood in AWS DeepRacer. This chapter provides definitions, analogies, and examples so you can build more effective RL models with AWS DeepRacer.



## So what does RL have to do with AWS DeepRacer?

AWS DeepRacer gives you the opportunity to apply machine learning in a fun way. But to take full advantage of the device, you need to figure out what’s going on behind the scenes. And for that, you need to take the time to understand reinforcement learning—a powerful and growing machine learning technique that literally drives AWS DeepRacer.

With reinforcement learning, models learn by a continuous process of receiving rewards and punishments for every action taken. It’s all about rewarding positive actions and punishing negative ones.

AWS DeepRacer’s interactions and movements around the track are entirely controlled by reinforcement learning. For instance, when AWS DeepRacer veers off the track, your model will learn, over time, to disincentivize that action so that it doesn't happen again. By contrast, when the device stays steady on the track, your model will learn to reward that action, encouraging it in the future.

Let’s look more closely at reinforcement learning so that you can more effectively use it to power AWS DeepRacer around the track.



# Machine Learning Algorithms



## Recap

- Supervised learning trains a model based on providing labels to each input, such as classifying between images of a bulldog and those that are not a bulldog.
- Unsupervised learning can use techniques like clustering to find relationships between different points of data, such as in detecting new types of fraud
- Reinforcement learning uses feedback from previous iterations, using trial and error, to improve



## Machine Learning Algorithms Further Research

This is all we will cover on Supervised and Unsupervised Learning for the rest of the course, as we’ll be focusing on Reinforcement Learning for AWS DeepRacer. If you want to learn more about these topics, check out the links below!

- [Supervised vs. Unsupervised Learning](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)
- [Intro to Machine Learning free course](https://www.udacity.com/course/intro-to-machine-learning--ud120)
- [Introduction to Machine Learning Nanodegree or Machine Learning Engineer Nanodegree programs](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)





# A Concrete Example

Let’s get concrete for a moment. Imagine a simple game wherein you have a robot that runs from left to right and must avoid obstacles by jumping over them at the right time. If your robot successfully clears an obstacle, it gets 1 point, but if the robot runs into it, it loses points. The goal of the game, of course, is for your robot to clear the obstacles and avoid a crash. Although simple in concept, this is a great starting point for understanding how reinforcement learning works and how you can use it to solve a concrete challenge.



[![RL agents can learn from experience, as well as explore new actions to gain better rewards.](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d1278ba_l3g1/l3g1.gif)RL agents can learn from experience, as well as explore new actions to gain better rewards.](https://classroom.udacity.com/courses/ud012/lessons/e13c8448-b27d-495c-a90f-f0604931aca7/concepts/63d1fda2-96d7-4e57-9c8c-b7beddf5f2c7#)



In applying reinforcement learning to this problem, you could first give the robot the ability to jump at various distances while approaching an obstacle. For instance, it could jump from 20 feet away, 10 feet away, 3 feet away, or even 0 feet away and crash into the obstacle before actually launching upwards. After a lot of trial and error experimenting with when to jump, your robot will eventually learn that it will receive a point by jumping just before hitting an oncoming obstacle (maybe that’s 3 feet away)—because that jumping-off point leads to a successful clearing of that obstacle. Your robot will also learn that it will quickly lose points by jumping too early (say, 20 feet away) or too late (say, 1 foot away) – as jumping-off points that are too far from or too close to an obstacle will lead to a crash. The reward and punishment, in the form of points, positively reinforces the action of jumping at that ideal point, eventually making your robot quite skilled at clearing the obstacles.

Of course, this game can get more complicated if the goal becomes, for instance, how quickly the robot can clear 10 obstacles in a row, or how well it can clear obstacles that require changing its direction. In this more complicated game, the robot could gain control over its speed and its ability to move in different directions.

By using reinforcement learning to help your robot succeed at this game the robot is now not only experimenting with different timings of its jumps, but the speed at which it runs and the specific direction of its jumps. Like the timing of jumps, the robot will receive rewards and punishments for the speeds and directions that result in more points and less points, respectively. Again, through a lot of trial and error, your robot will eventually learn the right mix of jumping time, speed, and direction to successfully clear a whole series of obstacles.







# RL Agents, Actions and States



We covered a lot in this video, so let’s summarize it briefly:

- Agent - the entity exhibiting certain behaviors (actions) based on its environment. In our case, it’s our AWS DeepRacer car.
- Actions - what the agent chooses to do at certain places in the environment, such as turning, going straight, going backward, etc. Actions can be discrete or continuous.
- States - has to do with where in the environment the agent resides (at a specific location) or with what is going on in the environment (for a robotic vacuum, perhaps that its current location is also clean). By taking actions, the agent moves from one state to a new state. States can be partial or absolute.





# State Transitions



To begin our cycle, the agent will choose an action given its starting state in the environment. It will then *transition* to a new state, where it will receive some reward. This will keep on in a continuous cycle of choosing a new action, moving to a new state, receiving a reward, and so on, for the rest of a given training episode.







# Discount Factors and Policies



- Discount factor - determines the extent to which future rewards should contribute to the overall sum of expected rewards. At a factor of zero, this means DeepRacer would only care about the very next action and its reward. With a factor of one, it pays attention to future rewards as well.

- Policy - this determines what action the agent takes given a particular state. Policies are split between stochastic and deterministic policies. Note that policy functions are often denoted by the symbol

   

  \pi*π*

  .

  - Stochastic - determines a probability for choosing a given action in a particular state (e.g. an 80% chance to go straight, 20% chance to turn left)
  - Deterministic - directly maps actions to states.

- Convolutional neural network - a neural network whose strength is determining some output from an image or set of images. In our case, it is used with the input image from the DeepRacer vehicle (whether real or simulated).



[![Our training cycle - the agent chooses an action, moving to a new state in the environment, and gaining some reward.](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d127901_l3g3/l3g3.png)Our training cycle - the agent chooses an action, moving to a new state in the environment, and gaining some reward.](https://classroom.udacity.com/courses/ud012/lessons/e13c8448-b27d-495c-a90f-f0604931aca7/concepts/930aebd8-ba25-4d98-8c5d-73c9db2c4629#)











# The Reward Function

In reinforcement learning, the reward function is the primary code used to incentivize optimal actions. It’s a mechanism used by the environment to let the agent know how it’s doing. The agent takes a particular action in a given state and receives either an immediate reward or a penalty.

For AWS DeepRacer, the reward function is vital to optimizing the models and enhancing performance around the track. For instance, when using the AWS DeepRacer console to train a model with a supported framework, the reward function is the only application-specific piece, and it depends on your input.

AWS DeepRacer includes various parameters that will help you optimize your reinforcement learning algorithm. Let’s take a look at those below before diving into our next video. **Click** on each parameter to learn more.



**all_wheels_on_track**

**Type:** Boolean 

**Syntax:** params['all_wheels_on_track'] 

**Description:** If all four wheels are on the track, where track is defined as the road surface including the border lines, then all_wheels_on_track is True. If any wheel is off the track, then all_wheels_on_track is False. Note: If all four wheels are off the track, the car will be reset. 



**x**

**Type:** Float 

**Syntax:** params['x'] 

**Description:** Returns the x coordinate of the center of the front axle of the car, in unit meters. 



**y**

**Type:** Float 

**Syntax:** params['y'] 

**Description:** Returns the y coordinate of the center of the front axle of the car, in unit meters. 



**distance_from_center**

**Type:** Float [0, track_width/2] 

**Syntax:** params['distance_from_center'] 

**Description:** Absolute distance from the center of the track. The center of the track is determined by the line that links all center waypoints. 



**is_left_of_center**

**Type:** Boolean 

**Syntax:** params['is_left_of_center'] 

**Description:** A variable that indicates if the car is to the left of the track's center. 



**is_reversed**

**Type:** Boolean 

**Syntax:** params['is_reversed'] 

**Description:** A variable that indicates whether the car is training in the original direction of the track or the reverse direction of the track. 



**heading**

**Type:** Float (-180,180] 

**Syntax:** params['heading'] 

**Description:** Returns the heading that the car is facing in degrees. When the car faces the direction of the x-axis increasing (with y constant), then it will return 0. When the car faces the direction of the y-axis increasing (with x constant), then it will return 90. When the car faces the direction of the y-axis decreasing (with x constant), then it will return -90. 



**progress**

**Type:** Float [0,100] 

**Syntax:** params['progress'] 

**Description:** Percentage of the track complete. A progress of 100 indicates that the lap is completed. 



**steps**

**Type:** Integer 

**Syntax:** params['steps'] 

**Description:** Number of steps completed. One step is one state, action, next state, reward tuple. 



**speed**

**Type:** Float 

**Syntax:** params['speed'] 

**Description:** The desired speed of the car in meters per second. This should match the selected action space. In other words, define this parameter within the limit that you set in the action space. 



**steering_angle**

**Type:** Float 

**Syntax:** params['steering_angle'] 

**Description:** The desired steering angle of the car in degrees. This should match the selected action space. (In other words, define this parameter within the limit that you set in the action space. Note that positive angles (+) indicate going left, and negative angles (-) indicate going right. This is aligned with 2D geometric processing. 



**track_width**

**Type:** Float 

**Syntax:** params['track_width'] 

**Description:** The width of the track, in unit meters. 



**waypoints**

**Type:** List 

**Syntax:** params['waypoints'] for the full list or params['waypoints'][i] for the i-th waypoint 

**Description:** Ordered list of waypoints that are spread around the center of the track, with each item in the list being the (x, y) coordinate of the waypoint. The list starts at zero. 



**closest_waypoints**

For an illustration of some of the parameters listed above, see the image below.



[![Examples of parameters on the track](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/June/5d13e2d5_l4-parameters/l4-parameters.png)Examples of parameters on the track](https://classroom.udacity.com/courses/ud012/lessons/0afdbee8-cf7a-4966-b8ed-f860f3d68cb6/concepts/4208eda1-1094-4e66-94d2-9450a7afc6f9#)







# A Basic Reward Function





In the video, Blaine shows a split between pre-made basic and advanced reward functions; in the current platform, the basic reward function is default while the others are available under *Reward function examples* in the console.

The basic reward function rewards the car staying near the center line. We can take a quick look back at our basic reward function code here:

```python
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    return float(reward)
```





# More Advanced Rewards





A best practice for iterating on reward functions is just to update one thing at a time - for instance, adding one additional parameter into the basic reward function. In Blaine’s updated function, he adds a parameter to avoid steering *too much* - hopefully, the car can go faster by going straighter more often than taking big back and forth turns around the center line.

Here is the code for the more advanced function:

```python
def reward_function(params):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''

    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the agent is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15

    # Penalize reward if the agent is steering too much
    if steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8

    return float(reward)
```







my hyper parameter 



5 4 3

| **Hyperparameter**                                           | **Value** |
| :----------------------------------------------------------- | :-------- |
| Gradient descent batch size                                  | 64        |
| Entropy                                                      | 0.01      |
| Discount factor                                              | 0.666     |
| Loss type                                                    | Huber     |
| Learning rate                                                | 0.0003    |
| Number of experience episodes between each policy-updating iteration | 20        |
| Number of epochs                                             | 10        |





​    

```python
`def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

# Read input parameters
track_width = params['track_width']
distance_from_center = params['distance_from_center']
all_wheels_on_track = params['all_wheels_on_track']
speed = params['speed']
SPEED_THRESHOLD = 1.0
# Calculate 3 markers that are at varying distances away from the center line
marker_1 = 0.1 * track_width
marker_2 = 0.25 * track_width
marker_3 = 0.5 * track_width

# Give higher reward if the car is closer to center line and vice versa
if distance_from_center <= marker_1:
    reward = 1.0
elif distance_from_center <= marker_2:
    reward = 0.5
elif distance_from_center <= marker_3:
    reward = 0.1
else:
    reward = 1e-3  # likely crashed/ close to off track
    
if not all_wheels_on_track:
	# Penalize if the car goes off track
    reward = 1e-3
elif speed < SPEED_THRESHOLD:
	# Penalize if the car goes too slow
    reward = reward + 0.5
else:
	# High reward if the car stays on track and goes fast
    reward = reward + 1.0r
    return float(reward)`
```
