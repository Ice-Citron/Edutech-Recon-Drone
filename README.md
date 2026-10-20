# Recon Drone

Waste detection in drone photos with YOLO11, a Flask backend and a React heatmap. A Marlborough College Malaysia team built it for the Planet Protectors Challenge at Google EDUtech Asia 2024 in Singapore. The team won first place.

[Project page](https://sienarindustries.com/project/edutech) ·
[Weights on Hugging Face](https://huggingface.co/shng2025/EduTech-YOLOv11) ·
[W&B report](https://forge.coreweave.com/wandb/marlborough-college-malaysia/Edutech/reports/Recon-Drone-MCM-Edutech-Asia--VmlldzoxODE0OTE0OA) ·
[Training report (PDF)](docs/recon-drone-training-report.pdf) ·
[Instagram](https://www.instagram.com/mcmedutech/)

<p align="center"><img src="docs/images/team-victory-by-stage.jpg" width="820" alt="The team by the stage after the win"></p>

## Videos

<table>
  <tr>
    <td width="33%"><a href="https://youtube.com/shorts/AQ2-JziMWtg"><img src="https://img.youtube.com/vi/AQ2-JziMWtg/maxresdefault.jpg" alt="Project video"></a></td>
    <td width="33%"><a href="https://youtu.be/nM7g6zC_H5I"><img src="https://img.youtube.com/vi/nM7g6zC_H5I/maxresdefault.jpg" alt="Finale team presentation"></a></td>
    <td width="33%"><a href="https://youtu.be/0D65uZaONBg"><img src="https://img.youtube.com/vi/0D65uZaONBg/maxresdefault.jpg" alt="Main-stage pitch"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Project video</b> (1 min)<br>Coastal litter detection with YOLO<br><a href="https://youtube.com/shorts/AQ2-JziMWtg">link</a></td>
    <td align="center"><b>Finale team presentation</b><br>Google EDUtech Asia 2024<br><a href="https://youtu.be/nM7g6zC_H5I">link</a></td>
    <td align="center"><b>Main-stage pitch</b><br>Marina Bay Sands Expo, Singapore<br><a href="https://youtu.be/0D65uZaONBg">link</a></td>
  </tr>
</table>

## How it works

A DJI drone flies a planned route and takes geotagged photos. The YOLO11 model finds waste in each photo and sorts it into eight classes. The web app plots the detections as a heatmap on Google Maps.

<p align="center"><img src="docs/images/webapp-heatmap.jpg" width="820" alt="Web app heatmap of detected waste on Google Maps"></p>

```
training/   dataset preparation, training scripts and Colab notebooks
webapp/     Flask backend and React front end (from Crispy2chickens/edutech)
docs/       training guide, report PDF, W&B charts and photos
```

## Results

The final model is YOLO11l after 9 epochs on one H100. It reaches mAP50 0.381 and mAP50-95 0.264 on validation. Runs, dataset and steps to reproduce are in [docs/training.md](docs/training.md).

<p align="center"><img src="docs/wandb/wandb-metrics.png" width="900" alt="mAP50, mAP50-95, precision and recall per epoch"></p>
<p align="center"><img src="docs/wandb/plots/sleek-dream-43/confusion.png" width="900" alt="Confusion matrix and normalised confusion matrix"></p>
<p align="center"><img src="docs/wandb/plots/sleek-dream-43/val_batch0_pred.jpg" width="640" alt="Predictions on one validation batch"></p>

## Use the model

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("shng2025/EduTech-YOLOv11", "weights/best.pt"))
model.predict("drone_photo.jpg", conf=0.25)[0].show()
```

## EDUtech Asia 2024

<table>
  <tr>
    <td width="25%"><img src="docs/images/finale-presentation.jpg" alt="Final presentation"></td>
    <td width="25%"><img src="docs/images/big-stage-speech-mbs.jpg" alt="Main-stage speech"></td>
    <td width="25%"><img src="docs/images/team-giving-demo.jpg" alt="Live demo"></td>
    <td width="25%"><img src="docs/images/shi-hao-and-jolie-demo.jpg" alt="Shi Hao and Jolie demo the web app"></td>
  </tr>
  <tr>
    <td><img src="docs/images/team-by-booth.jpg" alt="The team at the booth"></td>
    <td><img src="docs/images/exhibition-floor.jpg" alt="The exhibition floor"></td>
    <td><img src="docs/images/team-by-edutech-logo-chromebook.jpg" alt="With the prize Chromebooks"></td>
    <td><img src="docs/images/team-at-google-booth-chromebook.jpg" alt="At the Google booth"></td>
  </tr>
</table>

## Credits

- Shi Hao Ng ([Ice-Citron](https://github.com/Ice-Citron)): machine learning. Initial team lead.
- Jolie Teo ([Crispy2chickens](https://github.com/Crispy2chickens)): web app and backend. Team lead.
- Jonatas: web app.
- Louis Gan: drone and DJI SDK.

MIT License.
