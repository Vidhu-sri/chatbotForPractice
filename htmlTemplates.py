css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #475063;
    border-color: #FFAB2D;
}
.chat-message.bot {
    background-color: #2580D5
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIG.gUKnxIhk7lvy82.MKaZx?pid=ImgGn" style="max-height: 78px; max-width: 78px; border-radius: 5%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/GnananSai/CodeCraft-TM-PAI/main/static/person.png" style="border-radius: 5%;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''