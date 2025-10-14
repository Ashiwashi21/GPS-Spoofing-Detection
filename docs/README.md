GPS Spoofing Detection Project

Problem Statement:
How does a hybrid model of a Convolutional Neural Network (CNN) and an Extended Kalman Filter (EKF) compare to a CNN-only model when detecting GPS spoofing attacks accurately and efficiently?

Purpose:
The scientists chose this experiment because of their interest in aviation and coding to solve problems. Additionally, the sharp increase in GPS spoofing attacks targeting drones in recent years threatens aviation safety, and there is limited research on advanced models to detect spoofing attacks. By testing this project to find a more advanced approach to detecting attacks, this research aims to identify a faster and more reliable solution to this problem, and many agencies like the Department of Defense could apply this research in the real world.

Hypothesis:
If both a hybrid model of a 1D CNN with an EKF and a CNN-only model were trained to detect GPS spoofing attacks using injected coordinates, then the hybrid model will detect sophisticated attacks more accurately and efficiently because the EKF generates residuals for the CNN to detect anomalies, simplifying spoofing detection as the hybrid model doesn’t have to read the raw GPS data directly unlike the CNN-only model.

