import { Button } from "react-bootstrap";

export default function(props){
    return(
        <div style={{ display: "flex", alignItems: "center", width:"30rem" }}>
            <div style={{ flex: 1, backgroundColor: "lightgray", height: "2px" }} />
        
            <p style={{ margin: "0 10px", fontSize:"1.5rem" }}>{props.title}</p>
        
            <div style={{ flex: 1, backgroundColor: "lightgray", height: "2px", }} />
        </div>
    )
}






