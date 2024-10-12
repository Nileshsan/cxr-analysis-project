from django import forms

class XRayUploadForm(forms.Form):
    xray_image = forms.ImageField(label="Upload Chest X-ray")

from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()
