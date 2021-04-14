# Video Popularity Prediction Challenge
Prediction of the views of the videos based on their thumbnail images, meta data, tags and comments

About this Project:
The goal of this project is to develop a machine learning model to predict the number of views that videos are likely to receive based on attributes such as duration, location, and time published.

Information about the data: In order to build your machine learning model, we have been provided the following data sets:

1. **Metadata**

      comp_id: Unique ID

      ad_blocked: Indicates whether or not ads are blocked on the video

      embed: Indicates whether or not the video can be embedded

      ratio: The aspect ratio of the video

      duration: Duration of the video (in seconds)

      language: Language used in the video (encoded)

      partner: Indicates whether the video is certified by the partner/sponsor

      partner_active: Indicates whether the partner/sponsor is still active

      n_likes: The number of likes the video has

      n_tags: The number of tags in the video

      n_formats: The number of streaming formats available for the video

      dayofweek: The day of week when the video was published

      hour: The hour when the video was published (24-hour time format)


2. **Image data**
      Thumbnail pixel data


3. **Title data**
      Vectorized title data


4. **Description data**
      Vectorized descriptions


*Training datasets are marked with 'train_' at the beginning of their filenames. Please use these sets to develop the model. 
Datasets marked 'public_' at the beginning of their filenames are sets you can use to make predictions with your model and test how well your model performs on unseen data.

