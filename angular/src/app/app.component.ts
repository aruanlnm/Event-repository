import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'angular';
  text ;
  onEnter(value: string)
  {
    this.text = value;
    console.log(this.text);
  }

}
