import Gtk from 'gi://Gtk?version=4.0';

let app = new Gtk.Application({application_id: 'youxia.study.gnome.jshelloworld'});

app.connect('activate', () => {
    let win = new Gtk.ApplicationWindow({
        application: app,
        title: 'Hello World From JavaScript',
        default_width: 800,
        default_height: 600
    });
    let btn = new Gtk.Button({
        label: 'Hello, World!',
        valign: Gtk.Align.CENTER,
        halign: Gtk.Align.CENTER
    });
    btn.connect('clicked', () => {win.close();});
    win.set_child(btn);
    win.present();
});

app.run([]);
