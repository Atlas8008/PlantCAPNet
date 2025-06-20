function add_div(){
    var div = document.createElement("div");
    div.className = "background";
    document.getElementsByTagName("gradio-app")[0].appendChild(div);
    // document.body.appendChild(div);
}