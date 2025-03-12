# Use a base image with Flutter installed
FROM cirrusci/flutter:latest AS build

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Get Flutter dependencies
RUN flutter pub get

# Build the Flutter web project
RUN flutter build web

# Use Python image for the backend
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy the backend code
COPY --from=build /app /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the required port
EXPOSE 5000

# Start the backend
CMD ["python", "signlanguage/python_model/inference_classifier.py"]
