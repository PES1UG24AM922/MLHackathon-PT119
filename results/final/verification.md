# Independent release verification

- 31 automated tests passed, including synthetic evaluator mode and reference isolation.
- All ten student-test trips passed over live HTTP and downloaded Folium exports.
- Live point Top-1 and ordered-route F1 match the offline HMM run for every trip.
- The browser UI and the four-layer student report map rendered offline.
- The official evaluator dataset and physical two-laptop run remain reserved for the authorized demo.

After integrating the latest `main` decoder changes, all 33 tests passed.
The ten-trip API regression reproduced the recorded point and route metrics,
segment counts, and successful offline map exports.
