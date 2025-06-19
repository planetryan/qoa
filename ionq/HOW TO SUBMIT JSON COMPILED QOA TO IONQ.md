# HOW TO SUBMIT JSON COMPILED QOA TO IONQ

## STEP 1: COMPILE QOA TO JSON WITH:

cargo r compile-json /your/source/location/source.qoa /your/output/location/output.json

## STEP 2:

### CD into the output.json directory

## STEP 3: 

### Send to IonQ QPUs via curl:

If you havent setup your API key or QPU access, you would need to set it up in order to use IonQ

More about API Keys can be found here: https://docs.ionq.com/guides/managing-api-keys

### Curl template:

curl -X POST https://api.ionq.co/v0.2/jobs \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d @FILE_NAME_HERE.json

## STEP 4:

Check Job Status (Optional, can also be accessed via IonQ website
at https://cloud.ionq.com/jobs)

curl -X GET https://api.ionq.co/v0.2/jobs/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY_HERE"

## STEP 5:

Once the job is finished, the API response will include measurement results.

You can parse these results to analyze your circuit’s behavior (e.g., probability distribution of measurement outcomes).

## Notes:

"target": "simulator" runs your circuit on IonQ's simulator. Change to "quantum_computer" to run on real quantum hardware (but expect longer queue times).

"shots" controls how many measurement samples are taken. More shots give better statistics but take longer.

# Thank you for reading this guide, I hope it serves you well
### -- Rayan